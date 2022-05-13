/*******************************************************************************
    Copyright (c) 2018-2019 NVIDIA Corporation

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to
    deal in the Software without restriction, including without limitation the
    rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
    sell copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

        The above copyright notice and this permission notice shall be
        included in all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
    THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
    DEALINGS IN THE SOFTWARE.

*******************************************************************************/

#include "uvm_api.h"
#include "uvm_lock.h"
#include "uvm_kvmalloc.h"
#include "uvm_global.h"
#include "uvm_va_space.h"
#include "uvm_va_space_mm.h"
#include "uvm_ats_ibm.h"
#include "uvm_common.h"

#include <linux/pci.h>

#if UVM_IBM_NPU_SUPPORTED()

#include <linux/of.h>
#include <linux/sizes.h>
#include <asm/pci-bridge.h>
#include <asm/io.h>
#include <asm/iommu.h>
#include <asm/mmu.h>
#include <asm/mmu_context.h>
#include <asm/powernv.h>

#define NPU_ATSD_REG_MAP_SIZE 32

// There are three 8-byte registers in each ATSD mapping:
#define NPU_ATSD_REG_LAUNCH 0
#define NPU_ATSD_REG_AVA    1
#define NPU_ATSD_REG_STAT   2

// Fields within the NPU_ATSD_REG_LAUNCH register:

// "PRS" (process-scoped) bit. 1 means to limit invalidates to the specified
// PASID.
#define NPU_ATSD_REG_LAUNCH_PASID_ENABLE    13

// "PID" field. This specifies the PASID target of the invalidate.
#define NPU_ATSD_REG_LAUNCH_PASID_VAL       38

// "IS" bit. 0 means the specified virtual address range will be invalidated. 1
// means all entries will be invalidated.
#define NPU_ATSD_REG_LAUNCH_INVAL_ALL       12

// "AP" field. This encodes the size of a range-based invalidate.
#define NPU_ATSD_REG_LAUNCH_INVAL_SIZE      17

// "No flush" bit. 0 will trigger a flush (membar) from the GPU following the
// invalidate, 1 will not.
#define NPU_ATSD_REG_LAUNCH_FLUSH_DISABLE   39

// Helper to iterate over the active NPUs in the given VA space (all NPUs with
// GPUs that have GPU VA spaces registered in this VA space).
#define for_each_npu_index_in_va_space(npu_index, va_space)                                                 \
    for (({uvm_assert_rwlock_locked(&(va_space)->ats.ibm.rwlock);                                           \
           (npu_index) = find_first_bit((va_space)->ats.ibm.npu_active_mask, NV_MAX_NPUS);});               \
           (npu_index) < NV_MAX_NPUS;                                                                       \
           (npu_index) = find_next_bit((va_space)->ats.ibm.npu_active_mask, NV_MAX_NPUS, (npu_index) + 1))

// An invalidate requires operating on one set of registers in each NPU. This
// struct tracks which register set (id) is in use per NPU for a given
// operation.
typedef struct
{
    NvU8 ids[NV_MAX_NPUS];
} uvm_atsd_regs_t;

// Get the index of the input npu pointer within UVM's global npus array
static size_t uvm_ibm_npu_index(uvm_ibm_npu_t *npu)
{
    size_t npu_index = npu - &g_uvm_global.npus[0];
    UVM_ASSERT(npu_index < ARRAY_SIZE(g_uvm_global.npus));
    return npu_index;
}

// Find an existing NPU matching pci_domain, or return an empty NPU slot if none
// is found. Returns NULL if no slots are available.
static uvm_ibm_npu_t *uvm_ibm_npu_find(int pci_domain)
{
    size_t i;
    uvm_ibm_npu_t *npu, *first_free = NULL;

    uvm_assert_mutex_locked(&g_uvm_global.global_lock);

    for (i = 0; i < ARRAY_SIZE(g_uvm_global.npus); i++) {
        npu = &g_uvm_global.npus[i];
        if (npu->num_retained_gpus == 0) {
            if (!first_free)
                first_free = npu;
        }
        else if (npu->pci_domain == pci_domain) {
            return npu;
        }
    }

    return first_free;
}

static void uvm_ibm_npu_destroy(uvm_ibm_npu_t *npu)
{
    size_t i;

    uvm_assert_mutex_locked(&g_uvm_global.global_lock);
    UVM_ASSERT(npu->num_retained_gpus == 0);
    UVM_ASSERT(bitmap_empty(npu->atsd_regs.locks, UVM_MAX_ATSD_REGS));

    for (i = 0; i < npu->atsd_regs.count; i++) {
        UVM_ASSERT(npu->atsd_regs.io_addrs[i]);
        iounmap(npu->atsd_regs.io_addrs[i]);
    }

    memset(npu, 0, sizeof(*npu));
}

static NV_STATUS uvm_ibm_npu_init(uvm_ibm_npu_t *npu, struct pci_dev *npu_dev)
{
    struct pci_controller *hose;
    size_t i, reg_count, reg_size = sizeof(npu->atsd_regs.io_addrs[0]);
    int ret;

    uvm_assert_mutex_locked(&g_uvm_global.global_lock);
    UVM_ASSERT(npu->num_retained_gpus == 0);
    UVM_ASSERT(bitmap_empty(npu->atsd_regs.locks, UVM_MAX_ATSD_REGS));

    npu->pci_domain = pci_domain_nr(npu_dev->bus);

    if (!UVM_ATS_IBM_SUPPORTED_IN_DRIVER())
        return NV_OK;

    hose = pci_bus_to_host(npu_dev->bus);

    ret = of_property_count_elems_of_size(hose->dn, "ibm,mmio-atsd", reg_size);
    if (ret < 0) {
        UVM_ERR_PRINT("Failed to query NPU %d ATSD register count: %d\n", npu->pci_domain, ret);
        return errno_to_nv_status(ret);
    }

    // For ATS to be enabled globally, we must have NPU ATSD registers
    reg_count = ret;
    if (reg_count == 0 || reg_count > UVM_MAX_ATSD_REGS) {
        UVM_ERR_PRINT("NPU %d has invalid ATSD register count: %zu\n", npu->pci_domain, reg_count);
        return NV_ERR_INVALID_STATE;
    }

    // Map the ATSD registers
    for (i = 0; i < reg_count; i++) {
        u64 phys_addr;
        __be64 __iomem *io_addr;
        ret = of_property_read_u64_index(hose->dn, "ibm,mmio-atsd", i, &phys_addr);
        UVM_ASSERT(ret == 0);

        io_addr = ioremap(phys_addr, NPU_ATSD_REG_MAP_SIZE);
        if (!io_addr) {
            uvm_ibm_npu_destroy(npu);
            return NV_ERR_NO_MEMORY;
        }

        npu->atsd_regs.io_addrs[npu->atsd_regs.count++] = io_addr;
    }

    return NV_OK;
}

NV_STATUS uvm_ats_ibm_add_gpu(uvm_parent_gpu_t *parent_gpu)
{
    struct pci_dev *npu_dev = pnv_pci_get_npu_dev(parent_gpu->pci_dev, 0);
    uvm_ibm_npu_t *npu;
    NV_STATUS status;

    uvm_assert_mutex_locked(&g_uvm_global.global_lock);

    if (!npu_dev)
        return NV_OK;

    npu = uvm_ibm_npu_find(pci_domain_nr(npu_dev->bus));
    if (!npu) {
        // If this happens then we can't support the system configuation until
        // NV_MAX_NPUS is updated. Return the same error as when the number of
        // GPUs exceeds UVM_MAX_GPUS.
        UVM_ERR_PRINT("No more NPU slots available, update NV_MAX_NPUS\n");
        return NV_ERR_INSUFFICIENT_RESOURCES;
    }

    if (npu->num_retained_gpus == 0) {
        status = uvm_ibm_npu_init(npu, npu_dev);
        if (status != NV_OK)
            return status;
    }

    // This npu field could be read concurrently by a thread in the ATSD
    // invalidate path. We don't need to provide ordering with those threads
    // because those invalidates won't apply to the GPU being added until a GPU
    // VA space on this GPU is registered.
    npu->atsd_regs.num_membars = max(npu->atsd_regs.num_membars, parent_gpu->num_hshub_tlb_invalidate_membars);

    parent_gpu->npu = npu;
    ++npu->num_retained_gpus;
    return NV_OK;
}

void uvm_ats_ibm_remove_gpu(uvm_parent_gpu_t *parent_gpu)
{
    uvm_ibm_npu_t *npu = parent_gpu->npu;
    uvm_parent_gpu_t *other_parent_gpu;
    NvU32 num_membars_new = 0;

    uvm_assert_mutex_locked(&g_uvm_global.global_lock);

    if (!npu)
        return;

    UVM_ASSERT(npu->num_retained_gpus > 0);
    if (--npu->num_retained_gpus == 0) {
        uvm_ibm_npu_destroy(npu);
    }
    else {
        // Re-calculate the membar count
        for_each_parent_gpu(other_parent_gpu) {
            // The current GPU being removed should've already been removed from
            // the global list.
            UVM_ASSERT(other_parent_gpu != parent_gpu);
            if (other_parent_gpu->npu == npu)
                num_membars_new = max(num_membars_new, other_parent_gpu->num_hshub_tlb_invalidate_membars);
        }

        UVM_ASSERT(num_membars_new > 0);
        npu->atsd_regs.num_membars = num_membars_new;
    }
}

#if UVM_ATS_IBM_SUPPORTED()

void uvm_ats_ibm_init_va_space(uvm_va_space_t *va_space)
{
    uvm_ibm_va_space_t *ibm_va_space;

    UVM_ASSERT(va_space);
    ibm_va_space = &va_space->ats.ibm;

    uvm_rwlock_irqsave_init(&ibm_va_space->rwlock, UVM_LOCK_ORDER_LEAF);
}

#if UVM_ATS_IBM_SUPPORTED_IN_KERNEL()
static void npu_release_dummy(struct npu_context *npu_context, void *va_mm)
{
    // See the comment on the call to pnv_npu2_init_context()
}

static NV_STATUS uvm_ats_ibm_register_gpu_va_space_kernel(uvm_gpu_va_space_t *gpu_va_space)
{
    uvm_va_space_t *va_space = gpu_va_space->va_space;
    uvm_ibm_gpu_va_space_t *ibm_gpu_va_space = &gpu_va_space->ats.ibm;
    struct npu_context *npu_context;

    // pnv_npu2_init_context() registers current->mm with
    // mmu_notifier_register(). We need that to match the mm we passed to our
    // own mmu_notifier_register() for this VA space.
    if (current->mm != va_space->va_space_mm.mm)
        return NV_ERR_NOT_SUPPORTED;

    uvm_assert_mmap_lock_locked_write(current->mm);
    uvm_assert_rwsem_locked_write(&va_space->lock);

    // pnv_npu2_init_context() doesn't handle being called multiple times for
    // the same GPU under the same mm, which could happen if multiple VA spaces
    // are created in this process. To handle that we pass the VA space pointer
    // as the callback parameter: the callback values are shared by all devices
    // under this mm, so pnv_npu2_init_context() enforces that the values match
    // the ones already registered to the mm.
    //
    // Otherwise we don't use the callback, since we have our own callback
    // registered under the va_space_mm that will be called at the same point
    // (mmu_notifier release).
    npu_context = pnv_npu2_init_context(gpu_va_space->gpu->parent->pci_dev,
                                        (MSR_DR | MSR_PR | MSR_HV),
                                        npu_release_dummy,
                                        va_space);
    if (IS_ERR(npu_context)) {
        int err = PTR_ERR(npu_context);

        // We'll get -EINVAL if the callback value (va_space) differs from the
        // one already registered to the npu_context associated with this mm.
        // That can only happen when multiple VA spaces attempt registration
        // within the same process, which is disallowed and should return
        // NV_ERR_NOT_SUPPORTED.
        if (err == -EINVAL)
            return NV_ERR_NOT_SUPPORTED;
        return errno_to_nv_status(err);
    }

    ibm_gpu_va_space->npu_context = npu_context;

    return NV_OK;
}

static void uvm_ats_ibm_unregister_gpu_va_space_kernel(uvm_gpu_va_space_t *gpu_va_space)
{
    uvm_gpu_va_space_state_t state;
    uvm_va_space_t *va_space = gpu_va_space->va_space;
    uvm_ibm_va_space_t *ibm_va_space;
    uvm_ibm_gpu_va_space_t *ibm_gpu_va_space = &gpu_va_space->ats.ibm;

    if (!ibm_gpu_va_space->npu_context)
        return;

    // va_space is guaranteed to not be NULL if ibm_gpu_va_space->npu_context is
    // not NULL.
    UVM_ASSERT(va_space);

    state = uvm_gpu_va_space_state(gpu_va_space);
    UVM_ASSERT(state == UVM_GPU_VA_SPACE_STATE_INIT || state == UVM_GPU_VA_SPACE_STATE_DEAD);

    ibm_va_space = &va_space->ats.ibm;

    // pnv_npu2_destroy_context() may in turn call mmu_notifier_unregister().
    // If uvm_va_space_mm_shutdown() is concurrently executing in another
    // thread, mmu_notifier_unregister() will wait for
    // uvm_va_space_mm_shutdown() to finish. uvm_va_space_mm_shutdown() takes
    // mmap_lock and the VA space lock, so we can't be holding those locks on
    // this path.
    uvm_assert_unlocked_order(UVM_LOCK_ORDER_MMAP_LOCK);
    uvm_assert_unlocked_order(UVM_LOCK_ORDER_VA_SPACE);

    pnv_npu2_destroy_context(ibm_gpu_va_space->npu_context, gpu_va_space->gpu->parent->pci_dev);
    ibm_gpu_va_space->npu_context = NULL;
}

#else

static void uvm_ats_ibm_register_gpu_va_space_driver(uvm_gpu_va_space_t *gpu_va_space)
{
    uvm_va_space_t *va_space = gpu_va_space->va_space;
    uvm_ibm_gpu_va_space_t *ibm_gpu_va_space = &gpu_va_space->ats.ibm;
    uvm_gpu_t *gpu = gpu_va_space->gpu;
    size_t npu_index = uvm_ibm_npu_index(gpu->parent->npu);
    uvm_ibm_va_space_t *ibm_va_space;

    UVM_ASSERT(va_space);
    ibm_va_space = &va_space->ats.ibm;

    uvm_assert_rwsem_locked_write(&va_space->lock);

    uvm_write_lock_irqsave(&ibm_va_space->rwlock);

    // If this is the first GPU VA space to use this NPU in the VA space, mark
    // the NPU as active so invalidates are issued to it.
    if (ibm_va_space->npu_ref_counts[npu_index] == 0) {
        // If this is the first active NPU in the entire VA space, we have to
        // tell the kernel to send TLB invalidations to the IOMMU. See kernel
        // commit 03b8abedf4f4965e7e9e0d4f92877c42c07ce19f for background.
        //
        // This is safe to do without holding mm_users high or mmap_lock.
        if (bitmap_empty(ibm_va_space->npu_active_mask, NV_MAX_NPUS))
            mm_context_add_copro(va_space->va_space_mm.mm);

        UVM_ASSERT(!test_bit(npu_index, ibm_va_space->npu_active_mask));
        __set_bit(npu_index, ibm_va_space->npu_active_mask);
    }
    else {
        UVM_ASSERT(test_bit(npu_index, ibm_va_space->npu_active_mask));
    }

    ++ibm_va_space->npu_ref_counts[npu_index];

    // As soon as this lock is dropped, invalidates on this VA space's mm may
    // begin issuing ATSDs to this NPU.
    uvm_write_unlock_irqrestore(&ibm_va_space->rwlock);

    ibm_gpu_va_space->did_ibm_driver_init = true;
}

static void uvm_ats_ibm_unregister_gpu_va_space_driver(uvm_gpu_va_space_t *gpu_va_space)
{
    uvm_va_space_t *va_space = gpu_va_space->va_space;
    uvm_gpu_t *gpu = gpu_va_space->gpu;
    size_t npu_index = uvm_ibm_npu_index(gpu->parent->npu);
    bool do_remove = false;
    uvm_ibm_va_space_t *ibm_va_space;
    uvm_ibm_gpu_va_space_t *ibm_gpu_va_space = &gpu_va_space->ats.ibm;

    if (!ibm_gpu_va_space->did_ibm_driver_init)
        return;

    UVM_ASSERT(va_space);
    ibm_va_space = &va_space->ats.ibm;

    // Note that we aren't holding the VA space lock here, so another thread
    // could be in uvm_ats_ibm_register_gpu_va_space() for this same GPU right
    // now. The write lock and ref counts below will handle that case.

    // Once we return from this function with a bit cleared in the
    // npu_active_mask, we have to guarantee that this VA space no longer
    // accesses that NPU's ATSD registers. This is needed in case GPU unregister
    // needs to unmap those registers. We use the reader/writer lock to
    // guarantee this, which means that invalidations must not access the ATSD
    // registers outside of the lock.
    //
    // Future work: if we could synchronize_srcu() on the mmu_notifier SRCU we
    // might do that here instead to flush out all invalidates. That would allow
    // us to avoid taking a read lock in the invalidate path, though we'd have
    // to be careful when clearing the mask bit relative to the synchronize, and
    // we'd have to be careful in cases where this thread doesn't hold a
    // reference to mm_users.
    uvm_write_lock_irqsave(&ibm_va_space->rwlock);

    UVM_ASSERT(ibm_va_space->npu_ref_counts[npu_index] > 0);
    UVM_ASSERT(test_bit(npu_index, ibm_va_space->npu_active_mask));

    --ibm_va_space->npu_ref_counts[npu_index];
    if (ibm_va_space->npu_ref_counts[npu_index] == 0) {
        __clear_bit(npu_index, ibm_va_space->npu_active_mask);
        if (bitmap_empty(ibm_va_space->npu_active_mask, NV_MAX_NPUS))
            do_remove = true;
    }

    uvm_write_unlock_irqrestore(&ibm_va_space->rwlock);

    if (do_remove) {
        // mm_context_remove_copro() must be called outside of the spinlock
        // because it may issue invalidates across CPUs in this mm. The
        // coprocessor count is atomically refcounted by that function, so it's
        // safe to call here even if another thread jumps in with a register and
        // calls mm_context_add_copro() between this thread's unlock and this
        // call.
        UVM_ASSERT(va_space->va_space_mm.mm);
        mm_context_remove_copro(va_space->va_space_mm.mm);
    }
}

#endif // UVM_ATS_IBM_SUPPORTED_IN_KERNEL()

static mm_context_id_t va_space_pasid(uvm_va_space_t *va_space)
{
    struct mm_struct *mm = va_space->va_space_mm.mm;
    UVM_ASSERT(mm);
    return mm->context.id;
}

NV_STATUS uvm_ats_ibm_register_gpu_va_space(uvm_gpu_va_space_t *gpu_va_space)
{
    uvm_va_space_t *va_space = gpu_va_space->va_space;
    NV_STATUS status = NV_OK;

    UVM_ASSERT(gpu_va_space->ats.enabled);
    UVM_ASSERT(uvm_gpu_va_space_state(gpu_va_space) == UVM_GPU_VA_SPACE_STATE_INIT);
    UVM_ASSERT(va_space->va_space_mm.mm);
    uvm_assert_rwsem_locked_write(&va_space->lock);

#if UVM_ATS_IBM_SUPPORTED_IN_KERNEL()
    status = uvm_ats_ibm_register_gpu_va_space_kernel(gpu_va_space);
#else
    uvm_ats_ibm_register_gpu_va_space_driver(gpu_va_space);
#endif

    gpu_va_space->ats.pasid = (NvU32) va_space_pasid(gpu_va_space->va_space);

    return status;
}

void uvm_ats_ibm_unregister_gpu_va_space(uvm_gpu_va_space_t *gpu_va_space)
{
#if UVM_ATS_IBM_SUPPORTED_IN_KERNEL()
    uvm_ats_ibm_unregister_gpu_va_space_kernel(gpu_va_space);
#else
    uvm_ats_ibm_unregister_gpu_va_space_driver(gpu_va_space);
#endif

    gpu_va_space->ats.pasid = -1U;
}

#if UVM_ATS_IBM_SUPPORTED_IN_DRIVER()

// Find any available ATSD register set in this NPU and return that index. This
// will busy wait until a register set is free.
static NvU8 atsd_reg_acquire(uvm_ibm_npu_t *npu)
{
    uvm_spin_loop_t spin;
    size_t i;
    bool first = true;

    while (1) {
        // Using for_each_clear_bit is racy, since the bits could change at any
        // point. That's ok since we'll either just retry or use a real atomic
        // to lock the bit. Checking for clear bits first avoids spamming
        // atomics in the contended case.
        for_each_clear_bit(i, npu->atsd_regs.locks, npu->atsd_regs.count) {
            if (!test_and_set_bit_lock(i, npu->atsd_regs.locks))
                return (NvU8)i;
        }

        // Back off and try again, avoiding the overhead of initializing the
        // tracking timers unless we need them.
        if (first) {
            uvm_spin_loop_init(&spin);
            first = false;
        }
        else {
            UVM_SPIN_LOOP(&spin);
        }
    }
}

static void atsd_reg_release(uvm_ibm_npu_t *npu, NvU8 reg)
{
    UVM_ASSERT(reg < npu->atsd_regs.count);
    UVM_ASSERT(test_bit(reg, npu->atsd_regs.locks));
    clear_bit_unlock(reg, npu->atsd_regs.locks);
}

static __be64 atsd_reg_read(uvm_ibm_npu_t *npu, NvU8 reg, size_t offset)
{
    __be64 __iomem *io_addr = npu->atsd_regs.io_addrs[reg] + offset;
    UVM_ASSERT(reg < npu->atsd_regs.count);
    return __raw_readq(io_addr);
}

static void atsd_reg_write(uvm_ibm_npu_t *npu, NvU8 reg, size_t offset, NvU64 val)
{
    __be64 __iomem *io_addr = npu->atsd_regs.io_addrs[reg] + offset;
    UVM_ASSERT(reg < npu->atsd_regs.count);
    __raw_writeq_be(val, io_addr);
}

// Acquire a set of registers in each NPU which is active in va_space
static void atsd_regs_acquire(uvm_va_space_t *va_space, uvm_atsd_regs_t *regs)
{
    size_t i;
    for_each_npu_index_in_va_space(i, va_space)
        regs->ids[i] = atsd_reg_acquire(&g_uvm_global.npus[i]);
}

static void atsd_regs_release(uvm_va_space_t *va_space, uvm_atsd_regs_t *regs)
{
    size_t i;
    for_each_npu_index_in_va_space(i, va_space)
        atsd_reg_release(&g_uvm_global.npus[i], regs->ids[i]);
}

// Write the provided value to each NPU active in va_space at the provided
// register offset.
static void atsd_regs_write(uvm_va_space_t *va_space, uvm_atsd_regs_t *regs, size_t offset, NvU64 val)
{
    size_t i;
    for_each_npu_index_in_va_space(i, va_space)
        atsd_reg_write(&g_uvm_global.npus[i], regs->ids[i], offset, val);
}

// Wait for all prior operations issued to active NPUs in va_space on the given
// registers to finish.
static void atsd_regs_wait(uvm_va_space_t *va_space, uvm_atsd_regs_t *regs)
{
    uvm_spin_loop_t spin;
    size_t i;
    for_each_npu_index_in_va_space(i, va_space) {
        UVM_SPIN_WHILE(atsd_reg_read(&g_uvm_global.npus[i], regs->ids[i], NPU_ATSD_REG_STAT), &spin)
            ;
    }
}

// Encode an invalidate targeting the given pasid and the given size for the
// NPU_ATSD_REG_LAUNCH register. The target address is encoded separately.
//
// psize must be one of the MMU_PAGE_* values defined in powerpc's asm/mmu.h. A
// psize of MMU_PAGE_COUNT means to invalidate the entire address space.
static NvU64 atsd_get_launch_val(mm_context_id_t pasid, int psize)
{
    NvU64 val = 0;

    val |= PPC_BIT(NPU_ATSD_REG_LAUNCH_PASID_ENABLE);
    val |= pasid << PPC_BITLSHIFT(NPU_ATSD_REG_LAUNCH_PASID_VAL);

    if (psize == MMU_PAGE_COUNT) {
        val |= PPC_BIT(NPU_ATSD_REG_LAUNCH_INVAL_ALL);
    }
    else {
        // The NPU registers do not support arbitrary sizes
        UVM_ASSERT(psize == MMU_PAGE_64K || psize == MMU_PAGE_2M  || psize == MMU_PAGE_1G);
        val |= (NvU64)mmu_get_ap(psize) << PPC_BITLSHIFT(NPU_ATSD_REG_LAUNCH_INVAL_SIZE);
    }

    return val;
}

// Return the encoded size to use for an ATSD targeting the given range, in one
// of the MMU_PAGE_* values defined in powerpc's asm/mmu.h. A return value of
// MMU_PAGE_COUNT means the entire address space must be invalidated.
//
// start is an in/out parameter. On return start will be set to the aligned
// starting address to use for the ATSD. end is inclusive.
static int atsd_calc_size(NvU64 *start, NvU64 end)
{
    // ATSDs have high latency, so we prefer to over-invalidate rather than
    // issue multiple precise invalidates. Supported sizes are only 64K, 2M, and
    // 1G.

    *start = UVM_ALIGN_DOWN(*start, SZ_64K);
    end = UVM_ALIGN_DOWN(end, SZ_64K);
    if (*start == end)
        return MMU_PAGE_64K;

    *start = UVM_ALIGN_DOWN(*start, SZ_2M);
    end = UVM_ALIGN_DOWN(end, SZ_2M);
    if (*start == end)
        return MMU_PAGE_2M;

    *start = UVM_ALIGN_DOWN(*start, SZ_1G);
    end = UVM_ALIGN_DOWN(end, SZ_1G);
    if (*start == end)
        return MMU_PAGE_1G;

    return MMU_PAGE_COUNT;
}

// Issue an ATSD to all NPUs and wait for completion
static void atsd_launch_wait(uvm_va_space_t *va_space, uvm_atsd_regs_t *regs, NvU64 val)
{
    atsd_regs_write(va_space, regs, NPU_ATSD_REG_LAUNCH, val);
    atsd_regs_wait(va_space, regs);
}

// Issue and wait for the required membars following an invalidate
static void atsd_issue_membars(uvm_va_space_t *va_space, uvm_atsd_regs_t *regs)
{
    size_t i;
    NvU32 num_membars = 0;

    // These membars are issued using ATSDs which target a reserved PASID of 0.
    // That PASID is valid on the GPU in order for the membar to be valid, but
    // 0 will never be used by the kernel for an actual address space so the
    // ATSD won't actually invalidate any entries.
    NvU64 val = atsd_get_launch_val(0, MMU_PAGE_COUNT);

    for_each_npu_index_in_va_space(i, va_space) {
        uvm_ibm_npu_t *npu = &g_uvm_global.npus[i];
        num_membars = max(num_membars, npu->atsd_regs.num_membars);
    }

    for (i = 0; i < num_membars; i++)
        atsd_launch_wait(va_space, regs, val);
}

static void uvm_ats_ibm_invalidate_all(uvm_va_space_t *va_space, uvm_atsd_regs_t *regs)
{
    NvU64 val = atsd_get_launch_val(va_space_pasid(va_space), MMU_PAGE_COUNT);
    atsd_launch_wait(va_space, regs, val);
    atsd_issue_membars(va_space, regs);
}

static void uvm_ats_ibm_invalidate_range(uvm_va_space_t *va_space, uvm_atsd_regs_t *regs, NvU64 start, int psize)
{
    NvU64 val = atsd_get_launch_val(va_space_pasid(va_space), psize);

    // Barriers are expensive, so write all address registers first then do a
    // single barrier for all of them.
    atsd_regs_write(va_space, regs, NPU_ATSD_REG_AVA, start);
    eieio();
    atsd_launch_wait(va_space, regs, val);
    atsd_issue_membars(va_space, regs);
}

#endif // UVM_ATS_IBM_SUPPORTED_IN_DRIVER()

void uvm_ats_ibm_invalidate(uvm_va_space_t *va_space, NvU64 start, NvU64 end)
{
#if UVM_ATS_IBM_SUPPORTED_IN_DRIVER()
    unsigned long irq_flags;
    uvm_atsd_regs_t regs;
    NvU64 atsd_start = start;
    int psize = atsd_calc_size(&atsd_start, end);
    uvm_ibm_va_space_t *ibm_va_space = &va_space->ats.ibm;

    BUILD_BUG_ON(order_base_2(UVM_MAX_ATSD_REGS) > 8*sizeof(regs.ids[0]));

    // We must hold this lock in at least read mode when accessing NPU
    // registers. See the comment in uvm_ats_ibm_unregister_gpu_va_space_driver.
    uvm_read_lock_irqsave(&ibm_va_space->rwlock, irq_flags);

    if (!bitmap_empty(ibm_va_space->npu_active_mask, NV_MAX_NPUS)) {
        atsd_regs_acquire(va_space, &regs);

        if (psize == MMU_PAGE_COUNT)
            uvm_ats_ibm_invalidate_all(va_space, &regs);
        else
            uvm_ats_ibm_invalidate_range(va_space, &regs, atsd_start, psize);

        atsd_regs_release(va_space, &regs);
    }

    uvm_read_unlock_irqrestore(&ibm_va_space->rwlock, irq_flags);
#else
    UVM_ASSERT_MSG(0, "This function should not be called on this kernel version\n");
#endif // UVM_ATS_IBM_SUPPORTED_IN_DRIVER()
}

#if 0
/* Maximum number of nvlinks per npu */
#define NV_MAX_LINKS 6

struct npu_context {
	struct mm_struct *mm;
	struct pci_dev *npdev[NV_MAX_NPUS][NV_MAX_LINKS];
	struct mmu_notifier mn;
	struct kref kref;
	bool nmmu_flush;

	/* Callback to stop translation requests on a given GPU */
	void (*release_cb)(struct npu_context *context, void *priv);

	/*
	 * Private pointer passed to the above callback for usage by
	 * device drivers.
	 */
	void *priv;
};

struct mmio_atsd_reg {
	struct npu *npu;
	int reg;
};
//static DEFINE_SPINLOCK(npu_context_lock);


/*
 * NPU2 ATS
 */
/* Maximum possible number of ATSD MMIO registers per NPU */
#define NV_NMMU_ATSD_REGS 8
#define NV_NPU_MAX_PE_NUM	16

/*
 * A compound NPU IOMMU group which might consist of 1 GPU + 2xNPUs (POWER8) or
 * up to 3 x (GPU + 2xNPUs) (POWER9).
 */
struct npu_comp {
	struct iommu_table_group table_group;
	int pe_num;
	struct pnv_ioda_pe *pe[NV_NPU_MAX_PE_NUM];
};

/* An NPU descriptor, valid for POWER9 only */
struct npu {
	int index;
	__be64 *mmio_atsd_regs[NV_NMMU_ATSD_REGS];
	unsigned int mmio_atsd_count;

	/* Bitmask for MMIO register usage */
	unsigned long mmio_atsd_usage;

	/* Do we need to explicitly flush the nest mmu? */
	bool nmmu_flush;

	struct npu_comp npucomp;
};

/*
 * Find a free MMIO ATSD register and mark it in use. Return -ENOSPC
 * if none are available.
 */
static int get_mmio_atsd_reg(struct npu *npu)
{
	int i;

	for (i = 0; i < npu->mmio_atsd_count; i++) {
		if (!test_bit(i, &npu->mmio_atsd_usage))
			if (!test_and_set_bit_lock(i, &npu->mmio_atsd_usage))
				return i;
	}

	return -ENOSPC;
}

static void put_mmio_atsd_reg(struct npu *npu, int reg)
{
	clear_bit_unlock(reg, &npu->mmio_atsd_usage);
}

/* MMIO ATSD register offsets */
#define XTS_ATSD_LAUNCH 0
#define XTS_ATSD_AVA    1
#define XTS_ATSD_STAT   2

static unsigned long get_atsd_launch_val(unsigned long pid, unsigned long psize)
{
	unsigned long launch = 0;

	if (psize == MMU_PAGE_COUNT) {
		/* IS set to invalidate entire matching PID */
		launch |= PPC_BIT(12);
	} else {
		/* AP set to invalidate region of psize */
		launch |= (u64)mmu_get_ap(psize) << PPC_BITLSHIFT(17);
	}

	/* PRS set to process-scoped */
	launch |= PPC_BIT(13);

	/* PID */
	launch |= pid << PPC_BITLSHIFT(38);

	/* Leave "No flush" (bit 39) 0 so every ATSD performs a flush */

	return launch;
}
/* Maximum index of npu2 hosts in the system. Always < NV_MAX_NPUS */
static int max_npu2_index;

static void mmio_atsd_regs_write(struct mmio_atsd_reg
			mmio_atsd_reg[NV_MAX_NPUS], unsigned long offset,
			unsigned long val)
{
	struct npu *npu;
	int i, reg;

	for (i = 0; i <= max_npu2_index; i++) {
		reg = mmio_atsd_reg[i].reg;
		if (reg < 0)
			continue;

		npu = mmio_atsd_reg[i].npu;
		__raw_writeq_be(val, npu->mmio_atsd_regs[reg] + offset);
	}
}

static void mmio_invalidate_pid(struct mmio_atsd_reg mmio_atsd_reg[NV_MAX_NPUS],
				unsigned long pid)
{
	unsigned long launch = get_atsd_launch_val(pid, MMU_PAGE_COUNT);

	/* Invalidating the entire process doesn't use a va */
	mmio_atsd_regs_write(mmio_atsd_reg, XTS_ATSD_LAUNCH, launch);
}

static void mmio_invalidate_range(struct mmio_atsd_reg
			mmio_atsd_reg[NV_MAX_NPUS], unsigned long pid,
			unsigned long start, unsigned long psize)
{
	unsigned long launch = get_atsd_launch_val(pid, psize);

	/* Write all VAs first */
	mmio_atsd_regs_write(mmio_atsd_reg, XTS_ATSD_AVA, start);

	/* Issue one barrier for all address writes */
	eieio();

	/* Launch */
	mmio_atsd_regs_write(mmio_atsd_reg, XTS_ATSD_LAUNCH, launch);
}

#define mn_to_npu_context(x) container_of(x, struct npu_context, mn)

static void mmio_invalidate_wait(
	struct mmio_atsd_reg mmio_atsd_reg[NV_MAX_NPUS])
{
	struct npu *npu;
	int i, reg;

	/* Wait for all invalidations to complete */
	for (i = 0; i <= max_npu2_index; i++) {
		if (mmio_atsd_reg[i].reg < 0)
			continue;

		/* Wait for completion */
		npu = mmio_atsd_reg[i].npu;
		reg = mmio_atsd_reg[i].reg;
		while (__raw_readq(npu->mmio_atsd_regs[reg] + XTS_ATSD_STAT))
			cpu_relax();
	}
}

/*
 * Acquires all the address translation shootdown (ATSD) registers required to
 * launch an ATSD on all links this npu_context is active on.
 */
static void acquire_atsd_reg(struct npu_context *npu_context,
			struct mmio_atsd_reg mmio_atsd_reg[NV_MAX_NPUS])
{
	int i, j;
//	struct npu *npu;
	struct pci_dev *npdev;

	for (i = 0; i <= max_npu2_index; i++) {
		mmio_atsd_reg[i].reg = -1;
		for (j = 0; j < NV_MAX_LINKS; j++) {
			/*
			 * There are no ordering requirements with respect to
			 * the setup of struct npu_context, but to ensure
			 * consistent behaviour we need to ensure npdev[][] is
			 * only read once.
			 */
			npdev = READ_ONCE(npu_context->npdev[i][j]);
			if (!npdev)
				continue;
#if 0
            // FIXME
			npu = pci_bus_to_host(npdev->bus)->npu;
			if (!npu)
				continue;

			mmio_atsd_reg[i].npu = npu;
			mmio_atsd_reg[i].reg = get_mmio_atsd_reg(npu);
			while (mmio_atsd_reg[i].reg < 0) {
				mmio_atsd_reg[i].reg = get_mmio_atsd_reg(npu);
				cpu_relax();
			}
#endif
			break;
		}
	}
}

/*
 * Release previously acquired ATSD registers. To avoid deadlocks the registers
 * must be released in the same order they were acquired above in
 * acquire_atsd_reg.
 */
static void release_atsd_reg(struct mmio_atsd_reg mmio_atsd_reg[NV_MAX_NPUS])
{
	int i;

	for (i = 0; i <= max_npu2_index; i++) {
		/*
		 * We can't rely on npu_context->npdev[][] being the same here
		 * as when acquire_atsd_reg() was called, hence we use the
		 * values stored in mmio_atsd_reg during the acquire phase
		 * rather than re-reading npdev[][].
		 */
		if (mmio_atsd_reg[i].reg < 0)
			continue;

		put_mmio_atsd_reg(mmio_atsd_reg[i].npu, mmio_atsd_reg[i].reg);
	}
}

/*
 * Invalidate a virtual address range
 */
static void mmio_invalidate(struct npu_context *npu_context,
			unsigned long start, unsigned long size)
{
	struct mmio_atsd_reg mmio_atsd_reg[NV_MAX_NPUS];
	unsigned long pid = npu_context->mm->context.id;
	unsigned long atsd_start = 0;
	unsigned long end = start + size - 1;
	int atsd_psize = MMU_PAGE_COUNT;

	/*
	 * Convert the input range into one of the supported sizes. If the range
	 * doesn't fit, use the next larger supported size. Invalidation latency
	 * is high, so over-invalidation is preferred to issuing multiple
	 * invalidates.
	 *
	 * A 4K page size isn't supported by NPU/GPU ATS, so that case is
	 * ignored.
	 */
	if (size == SZ_64K) {
		atsd_start = start;
		atsd_psize = MMU_PAGE_64K;
	} else if (ALIGN_DOWN(start, SZ_2M) == ALIGN_DOWN(end, SZ_2M)) {
		atsd_start = ALIGN_DOWN(start, SZ_2M);
		atsd_psize = MMU_PAGE_2M;
	} else if (ALIGN_DOWN(start, SZ_1G) == ALIGN_DOWN(end, SZ_1G)) {
		atsd_start = ALIGN_DOWN(start, SZ_1G);
		atsd_psize = MMU_PAGE_1G;
	}

	if (npu_context->nmmu_flush)
		/*
		 * Unfortunately the nest mmu does not support flushing specific
		 * addresses so we have to flush the whole mm once before
		 * shooting down the GPU translation.
		 */
		flush_all_mm(npu_context->mm);

	/*
	 * Loop over all the NPUs this process is active on and launch
	 * an invalidate.
	 */
	acquire_atsd_reg(npu_context, mmio_atsd_reg);

	if (atsd_psize == MMU_PAGE_COUNT)
		mmio_invalidate_pid(mmio_atsd_reg, pid);
	else
		mmio_invalidate_range(mmio_atsd_reg, pid, atsd_start,
					atsd_psize);

	mmio_invalidate_wait(mmio_atsd_reg);

	/*
	 * The GPU requires two flush ATSDs to ensure all entries have been
	 * flushed. We use PID 0 as it will never be used for a process on the
	 * GPU.
	 */
	mmio_invalidate_pid(mmio_atsd_reg, 0);
	mmio_invalidate_wait(mmio_atsd_reg);
	mmio_invalidate_pid(mmio_atsd_reg, 0);
	mmio_invalidate_wait(mmio_atsd_reg);

	release_atsd_reg(mmio_atsd_reg);
}

static void pnv_npu2_mn_release(struct mmu_notifier *mn,
				struct mm_struct *mm)
{
	struct npu_context *npu_context = mn_to_npu_context(mn);

	/* Call into device driver to stop requests to the NMMU */
	if (npu_context->release_cb)
		npu_context->release_cb(npu_context, npu_context->priv);

	/*
	 * There should be no more translation requests for this PID, but we
	 * need to ensure any entries for it are removed from the TLB.
	 */
	mmio_invalidate(npu_context, 0, ~0UL);
}

static void pnv_npu2_mn_invalidate_range(struct mmu_notifier *mn,
					struct mm_struct *mm,
					unsigned long start, unsigned long end)
{
	struct npu_context *npu_context = mn_to_npu_context(mn);
	mmio_invalidate(npu_context, start, end - start);
}

static const struct mmu_notifier_ops nv_nmmu_notifier_ops = {
	.release = pnv_npu2_mn_release,
	.invalidate_range = pnv_npu2_mn_invalidate_range,
};

struct npu_context *pnv_npu2_init_context(struct pci_dev *gpdev,
			unsigned long flags,
			void (*cb)(struct npu_context *, void *),
			void *priv)
{
	int rc;
	u32 nvlink_index;
	struct device_node *nvlink_dn;
	struct mm_struct *mm = current->mm;
	struct npu *npu;
	struct npu_context *npu_context;
	struct pci_controller *hose;

	/*
	 * At present we don't support GPUs connected to multiple NPUs and I'm
	 * not sure the hardware does either.
	 */
	struct pci_dev *npdev = pnv_pci_get_npu_dev(gpdev, 0);

	if (!npdev)
		/* No nvlink associated with this GPU device */
		return ERR_PTR(-ENODEV);

	/* We only support DR/PR/HV in pnv_npu2_map_lpar_dev() */
	if (flags & ~(MSR_DR | MSR_PR | MSR_HV))
		return ERR_PTR(-EINVAL);

	nvlink_dn = of_parse_phandle(npdev->dev.of_node, "ibm,nvlink", 0);
	if (WARN_ON(of_property_read_u32(nvlink_dn, "ibm,npu-link-index",
							&nvlink_index)))
		return ERR_PTR(-ENODEV);

	if (!mm || mm->context.id == 0) {
		/*
		 * Kernel thread contexts are not supported and context id 0 is
		 * reserved on the GPU.
		 */
		return ERR_PTR(-EINVAL);
	}

	hose = pci_bus_to_host(npdev->bus);
#if 0
	npu = hose->npu;
	if (!npu)
		return ERR_PTR(-ENODEV);

	/*
	 * We store the npu pci device so we can more easily get at the
	 * associated npus.
	 */
	spin_lock(&npu_context_lock);
	npu_context = mm->context.npu_context;
	if (npu_context) {
		if (npu_context->release_cb != cb ||
			npu_context->priv != priv) {
			spin_unlock(&npu_context_lock);
			return ERR_PTR(-EINVAL);
		}

		WARN_ON(!kref_get_unless_zero(&npu_context->kref));
	}
	spin_unlock(&npu_context_lock);
#endif

	if (!npu_context) {
		/*
		 * We can set up these fields without holding the
		 * npu_context_lock as the npu_context hasn't been returned to
		 * the caller meaning it can't be destroyed. Parallel allocation
		 * is protected against by mmap_sem.
		 */
		rc = -ENOMEM;
		npu_context = kzalloc(sizeof(struct npu_context), GFP_KERNEL);
		if (npu_context) {
			kref_init(&npu_context->kref);
			npu_context->mm = mm;
			npu_context->mn.ops = &nv_nmmu_notifier_ops;
			rc = __mmu_notifier_register(&npu_context->mn, mm);
		}

		if (rc) {
			kfree(npu_context);
			return ERR_PTR(rc);
		}

//		mm->context.npu_context = npu_context;
	}

	npu_context->release_cb = cb;
	npu_context->priv = priv;

	/*
	 * npdev is a pci_dev pointer setup by the PCI code. We assign it to
	 * npdev[][] to indicate to the mmu notifiers that an invalidation
	 * should also be sent over this nvlink. The notifiers don't use any
	 * other fields in npu_context, so we just need to ensure that when they
	 * deference npu_context->npdev[][] it is either a valid pointer or
	 * NULL.
	 */
	WRITE_ONCE(npu_context->npdev[npu->index][nvlink_index], npdev);

	if (!npu->nmmu_flush) {
		/*
		 * If we're not explicitly flushing ourselves we need to mark
		 * the thread for global flushes
		 */
		npu_context->nmmu_flush = false;
		mm_context_add_copro(mm);
	} else
		npu_context->nmmu_flush = true;

	return npu_context;
}
EXPORT_SYMBOL(pnv_npu2_init_context);

static void pnv_npu2_release_context(struct kref *kref)
{
	struct npu_context *npu_context =
		container_of(kref, struct npu_context, kref);

	if (!npu_context->nmmu_flush)
		mm_context_remove_copro(npu_context->mm);

//	npu_context->mm->context.npu_context = NULL;
}


static struct pci_dev *get_pci_dev(struct device_node *dn)
{
	struct pci_dn *pdn = PCI_DN(dn);
	struct pci_dev *pdev;

	pdev = pci_get_domain_bus_and_slot(pci_domain_nr(pdn->phb->bus),
					   pdn->busno, pdn->devfn);

	/*
	 * pci_get_domain_bus_and_slot() increased the reference count of
	 * the PCI device, but callers don't need that actually as the PE
	 * already holds a reference to the device. Since callers aren't
	 * aware of the reference count change, call pci_dev_put() now to
	 * avoid leaks.
	 */
	if (pdev)
		pci_dev_put(pdev);

	return pdev;
}

/* Given a NPU device get the associated PCI device. */
struct pci_dev *pnv_pci_get_gpu_dev(struct pci_dev *npdev)
{
	struct device_node *dn;
	struct pci_dev *gpdev;

	if (WARN_ON(!npdev))
		return NULL;

	if (WARN_ON(!npdev->dev.of_node))
		return NULL;

	/* Get assoicated PCI device */
	dn = of_parse_phandle(npdev->dev.of_node, "ibm,gpu", 0);
	if (!dn)
		return NULL;

	gpdev = get_pci_dev(dn);
	of_node_put(dn);

	return gpdev;
}
EXPORT_SYMBOL(pnv_pci_get_gpu_dev);

/* Given the real PCI device get a linked NPU device. */
struct pci_dev *pnv_pci_get_npu_dev(struct pci_dev *gpdev, int index)
{
	struct device_node *dn;
	struct pci_dev *npdev;

	if (WARN_ON(!gpdev))
		return NULL;

	/* Not all PCI devices have device-tree nodes */
	if (!gpdev->dev.of_node)
		return NULL;

	/* Get assoicated PCI device */
	dn = of_parse_phandle(gpdev->dev.of_node, "ibm,npu", index);
	if (!dn)
		return NULL;

	npdev = get_pci_dev(dn);
	of_node_put(dn);

	return npdev;
}
EXPORT_SYMBOL(pnv_pci_get_npu_dev);
#endif

#endif // UVM_ATS_IBM_SUPPORTED
#endif // UVM_IBM_NPU_SUPPORTED
