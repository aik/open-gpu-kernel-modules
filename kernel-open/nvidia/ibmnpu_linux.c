/* _NVRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2014-2019 by NVIDIA Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 *
 * _NVRM_COPYRIGHT_END_
 */

#include <linux/module.h>
#include <linux/interrupt.h>
#include "conftest.h"
#include "nv-pci-types.h"

#include "nvlink_pci.h"
#include "nvlink_proto.h"
#include "nvlink_os.h"
#include "nvlink_errors.h"

#include "ibmnpu_export.h"
#include "ibmnpu_linux.h"
#ifndef IRQ_RETVAL
typedef void irqreturn_t;
#define IRQ_RETVAL(a)
#endif

#if !defined(IRQF_SHARED)
#define IRQF_SHARED SA_SHIRQ
#endif

static int              ibmnpu_probe    (struct pci_dev *, const struct pci_device_id *);
static void             ibmnpu_remove   (struct pci_dev *);
static pci_ers_result_t ibmnpu_pci_error_detected   (struct pci_dev *, nv_pci_channel_state_t);
static pci_ers_result_t ibmnpu_pci_mmio_enabled     (struct pci_dev *);

static struct pci_error_handlers ibmnpu_pci_error_handlers =
{
    .error_detected = ibmnpu_pci_error_detected,
    .mmio_enabled   = ibmnpu_pci_mmio_enabled,
};

static struct pci_device_id ibmnpu_pci_table[] = {
    {
        .vendor      = PCI_VENDOR_ID_IBM,
        .device      = PCI_DEVICE_ID_IBM_NPU,
        .subvendor   = PCI_ANY_ID,
        .subdevice   = PCI_ANY_ID,
        .class       = (PCI_CLASS_BRIDGE_NPU << 8),
        .class_mask  = ~1
    },
    { }
};

static struct pci_driver ibmnpu_pci_driver = {
    .name           = IBMNPU_DRIVER_NAME,
    .id_table       = ibmnpu_pci_table,
    .probe          = ibmnpu_probe,
    .remove         = ibmnpu_remove,
    .err_handler    = &ibmnpu_pci_error_handlers,
};

/* Low-priority, preemptible watchdog for checking device status */
static struct {
    struct mutex        lock;
    struct delayed_work work;
    struct list_head    devices;
    NvBool              rearm;
} g_ibmnpu_watchdog;

typedef struct {
    struct pci_dev  *dev;
    struct list_head list_node;
} ibmnpu_watchdog_device;

static void ibmnpu_watchdog_check_devices(struct work_struct *);

static void ibmnpu_init_watchdog(void)
{
    mutex_init(&g_ibmnpu_watchdog.lock);
    INIT_DELAYED_WORK(&g_ibmnpu_watchdog.work, ibmnpu_watchdog_check_devices);
    INIT_LIST_HEAD(&g_ibmnpu_watchdog.devices);
    g_ibmnpu_watchdog.rearm = NV_TRUE;
}

static void ibmnpu_shutdown_watchdog(void)
{
    struct list_head *cur, *next;

    mutex_lock(&g_ibmnpu_watchdog.lock);

    g_ibmnpu_watchdog.rearm = NV_FALSE;

    mutex_unlock(&g_ibmnpu_watchdog.lock);

    /*
     * Wait to make sure the watchdog finishes its execution before proceeding
     * with the teardown.
     */
    flush_delayed_work(&g_ibmnpu_watchdog.work);

    mutex_lock(&g_ibmnpu_watchdog.lock);

    /*
     * Remove any remaining devices in the watchdog's check list (although
     * they should already have been removed in the typical case).
     */
    if (!list_empty(&g_ibmnpu_watchdog.devices))
    {
        nvlink_print(NVLINK_DBG_ERRORS,
            "IBMNPU: watchdog still running on devices:\n");
        list_for_each_safe(cur, next, &g_ibmnpu_watchdog.devices)
        {
            ibmnpu_watchdog_device *wd_dev =
                list_entry(cur, ibmnpu_watchdog_device, list_node);

            nvlink_print(NVLINK_DBG_ERRORS, "IBMNPU:    " NV_PCI_DEV_FMT "\n",
                NV_PCI_DEV_FMT_ARGS(wd_dev->dev));

            list_del(cur);
            kfree(wd_dev);
        }
    }

    mutex_unlock(&g_ibmnpu_watchdog.lock);
}

/*
 * Add a device to the list of devices that the watchdog will periodically
 * check on. Start the watchdog if this is the first device to be registered.
 */
static NvlStatus ibmnpu_start_watchdog_device(struct pci_dev *dev)
{
    NvlStatus retval = NVL_SUCCESS;
    ibmnpu_watchdog_device *wd_dev;

    mutex_lock(&g_ibmnpu_watchdog.lock);

    wd_dev = kmalloc(sizeof(ibmnpu_watchdog_device), GFP_KERNEL);
    if (wd_dev != NULL)
    {
        wd_dev->dev = dev;
        list_add_tail(&wd_dev->list_node, &g_ibmnpu_watchdog.devices);
        if (list_is_singular(&g_ibmnpu_watchdog.devices))
        {
            /* Make the watchdog work item re-schedule itself */
            g_ibmnpu_watchdog.rearm = NV_TRUE;
            schedule_delayed_work(&g_ibmnpu_watchdog.work, HZ);
        }
    }
    else
    {
        nvlink_print(NVLINK_DBG_ERRORS,
            "IBMNPU: failed to allocate watchdog state for device "
            NV_PCI_DEV_FMT "\n", NV_PCI_DEV_FMT_ARGS(dev));
        retval = -NVL_NO_MEM;
    }

    mutex_unlock(&g_ibmnpu_watchdog.lock);

    return retval;
}

/*
 * Stops the watchdog from checking the given device and waits for the
 * watchdog to finish, if no more devices need to be check.
 */
static void ibmnpu_stop_watchdog_device(struct pci_dev *dev)
{
    struct list_head *cur;

    mutex_lock(&g_ibmnpu_watchdog.lock);

    list_for_each(cur, &g_ibmnpu_watchdog.devices)
    {
        ibmnpu_watchdog_device *wd_dev =
            list_entry(cur, ibmnpu_watchdog_device, list_node);

        if (wd_dev->dev == dev)
        {
            list_del(cur);
            kfree(wd_dev);
            break;
        }
    }

    g_ibmnpu_watchdog.rearm = !list_empty(&g_ibmnpu_watchdog.devices);

    mutex_unlock(&g_ibmnpu_watchdog.lock);

    if (!g_ibmnpu_watchdog.rearm)
    {
        /*
         * Wait for the last work item to complete before proceeding with
         * the teardown. We must not hold the lock here so that the watchdog
         * work item can proceed.
         */
        flush_delayed_work(&g_ibmnpu_watchdog.work);
    }
}

/*
 * Periodic callback to check NPU devices for failure.
 *
 * This executes as a work item that re-schedules itself.
 */
static void ibmnpu_watchdog_check_devices
(
    struct work_struct * __always_unused work
)
{
    struct list_head *cur, *next;

    mutex_lock(&g_ibmnpu_watchdog.lock);

    list_for_each_safe(cur, next, &g_ibmnpu_watchdog.devices)
    {
        ibmnpu_watchdog_device *wd_dev =
            list_entry(cur, ibmnpu_watchdog_device, list_node);

        if (unlikely(ibmnpu_os_device_check_failure(wd_dev->dev)))
        {
            /*
             * Mark the device as failed, and remove it from the watchdog's
             * check list. No need to print anything, since the EEH handler
             * ibmnpu_pci_error_detected() will have already been run for this
             * device.
             */
            list_del(cur);
            kfree(wd_dev);
        }
    }

    /*
     * Stop the watchdog from rescheduling itself if there are no more
     * devices left to check on.
     */
    if (unlikely(list_empty(&g_ibmnpu_watchdog.devices)))
    {
        g_ibmnpu_watchdog.rearm = NV_FALSE;
    }
    else if (likely(g_ibmnpu_watchdog.rearm))
    {
        schedule_delayed_work(&g_ibmnpu_watchdog.work, HZ);
    }

    mutex_unlock(&g_ibmnpu_watchdog.lock);
}

static irqreturn_t ibmnpu_isr
(
    int   irq,
    void *arg
)
{
    NvlStatus retval;
    nvlink_pci_info *info = (nvlink_pci_info *)arg;

    if (NULL == arg)
    {
        nvlink_print(NVLINK_DBG_ERRORS,
            "An interrupt was fired for an NPU device, but no device info was "
            "provided\n");
        return IRQ_NONE;
    }

    /*
     * ibmnpu_lib_service_device returns a NOT_FOUND error code when it
     * couldn't find an interrupt to service. Occasionally we can get
     * spurious interrupts when interrupts are disabled, so this will
     * silence the prints for that case.
     */
    retval = ibmnpu_lib_service_device(info);
    if (-NVL_NOT_FOUND != retval)
    {
        nvlink_print(NVLINK_DBG_ERRORS,
            "IBMNPU: An interrupt has occurred on NPU device " NV_PCI_DEV_FMT
            " (%x)\n",
            info->domain, info->bus, info->device, info->function, retval);
    }

    return IRQ_HANDLED;
}

static void ibmnpu_brick_unload_pci_bar_info(ibmnpu_brick_info_t *brick_info)
{
    unsigned int bar;

    for (bar = 0; bar < IBMNPU_MAX_BARS; bar++)
    {
        struct nvlink_pci_bar_info *bar_info =
            &brick_info->pci_info.bars[bar];
        if (NULL != bar_info->pBar)
        {
            pci_iounmap(brick_info->dev, bar_info->pBar);
            bar_info->pBar = NULL;
        }
    }

    pci_release_regions(brick_info->dev);
}

static int ibmnpu_brick_load_pci_bar_info(ibmnpu_brick_info_t *brick_info)
{
    int rc;
    unsigned int i, j;
    struct pci_dev *dev = brick_info->dev;

    rc = pci_request_regions(dev, IBMNPU_DRIVER_NAME);
    if (rc != 0)
    {
        nvlink_print(NVLINK_DBG_ERRORS,
            "IBMNPU: Failed to request memory regions (%d)\n", rc);
        return rc;
    }

    for (i = 0, j = 0; i < NVRM_PCICFG_NUM_BARS && j < IBMNPU_MAX_BARS; i++)
    {
        if ((NV_PCI_RESOURCE_VALID(dev, i)) &&
            ((NV_PCI_RESOURCE_FLAGS(dev, i) & PCI_BASE_ADDRESS_SPACE) ==
                PCI_BASE_ADDRESS_SPACE_MEMORY))
        {
            unsigned int bar;
            struct nvlink_pci_bar_info *bar_info =
                &brick_info->pci_info.bars[j];

            bar_info->offset = NVRM_PCICFG_BAR_OFFSET(i);
            pci_read_config_dword(dev, bar_info->offset, &bar);
            bar_info->busAddress = (bar & PCI_BASE_ADDRESS_MEM_MASK);
            if (NV_PCI_RESOURCE_FLAGS(dev, i) & PCI_BASE_ADDRESS_MEM_TYPE_64)
            {
                pci_read_config_dword(dev, bar_info->offset + 4, &bar);
                bar_info->busAddress |= (((NvU64)bar) << 32);
            }

            bar_info->baseAddr = NV_PCI_RESOURCE_START(dev, i);
            bar_info->barSize = NV_PCI_RESOURCE_SIZE(dev, i);

            nvlink_print(NVLINK_DBG_INFO,
                "IBMNPU: BAR%d @ 0x%llx [size=%dK].\n",
                j, bar_info->baseAddr, (bar_info->barSize >> 10));

            /* Map registers to kernel address space. */
            bar_info->pBar = pci_iomap(dev, i, 0);
            if (NULL == bar_info->pBar)
            {
                nvlink_print(NVLINK_DBG_ERRORS,
                    "IBMNPU: Unable to map BAR%d registers\n", j);
                ibmnpu_brick_unload_pci_bar_info(brick_info);
                return -1;
            }
            j++;
        }
    }

    return 0;
}

static int ibmnpu_device_load_brick_info(struct pci_dev *dev)
{
    int rc;
    ibmnpu_brick_info_t *brick_info;

    brick_info = kzalloc(sizeof(*brick_info), GFP_KERNEL);
    if (!brick_info)
    {
        return -ENOMEM;
    }

    brick_info->dev = dev;

    rc = ibmnpu_brick_load_pci_bar_info(brick_info);
    if (rc)
    {
        nvlink_print(NVLINK_DBG_ERRORS,
            "IBMNPU: Failed to load device BAR info (%d)\n", rc);
        kfree(brick_info);
        return rc;
    }

    if (NVL_SUCCESS != ibmnpu_start_watchdog_device(dev))
    {
        ibmnpu_brick_unload_pci_bar_info(brick_info);
        kfree(brick_info);
        return -1;
    }

    pci_set_drvdata(dev, brick_info);

    return 0;
}

static void ibmnpu_device_unload_brick_info(struct pci_dev *dev)
{
    ibmnpu_brick_info_t *brick_info = pci_get_drvdata(dev);

    pci_set_drvdata(dev, NULL);
    ibmnpu_stop_watchdog_device(dev);
    ibmnpu_brick_unload_pci_bar_info(brick_info);
    kfree(brick_info);
}

static int ibmnpu_probe
(
    struct pci_dev *dev,
    const struct pci_device_id *id_table
)
{
    int rc;

    nvlink_print(NVLINK_DBG_SETUP,
        "IBMNPU: Probing Emulated device " NV_PCI_DEV_FMT ", "
        "Vendor Id = 0x%x, Device Id = 0x%x, Class = 0x%x \n",
        NV_PCI_DEV_FMT_ARGS(dev), dev->vendor, dev->device, dev->class);

    rc = pci_enable_device(dev);
    if (0 != rc)
    {
        nvlink_print(NVLINK_DBG_ERRORS,
            "IBMNPU: Failed to enable pci device (%d)\n", rc);
        return rc;
    }

    // Enable bus mastering on the device
    pci_set_master(dev);

    rc = ibmnpu_device_load_brick_info(dev);
    if (rc)
    {
        nvlink_print(NVLINK_DBG_ERRORS,
            "IBMNPU: Failed to load brick info for device "
            NV_PCI_DEV_FMT " (%d)\n", NV_PCI_DEV_FMT_ARGS(dev), rc);
        pci_disable_device(dev);
        return rc;
    }

    if (dev->irq == 0)
    {
        nvlink_print(NVLINK_DBG_ERRORS,
            "IBMNPU: Can't find an IRQ!\n");
        ibmnpu_remove(dev);
        return -EIO;
    }

    return 0;
}

void ibmnpu_remove(struct pci_dev *dev)
{
    nvlink_print(NVLINK_DBG_SETUP,
        "IBMNPU: Removing device " NV_PCI_DEV_FMT "\n",
        NV_PCI_DEV_FMT_ARGS(dev));

    ibmnpu_device_unload_brick_info(dev);
    pci_disable_device(dev);
}

static pci_ers_result_t ibmnpu_pci_error_detected
(
    struct pci_dev *dev,
    nv_pci_channel_state_t error
)
{
    ibmnpu_brick_info_t *brick_info;

    if (NULL == dev)
    {
        return PCI_ERS_RESULT_NONE;
    }

    brick_info = pci_get_drvdata(dev);

    nvlink_print(NVLINK_DBG_ERRORS,
        "IBMNPU: ibmnpu_pci_error_detected device " NV_PCI_DEV_FMT "\n",
        NV_PCI_DEV_FMT_ARGS(dev));

    // Mark the device as off-limits
    ibmnpu_lib_stop_device_mmio(&brick_info->pci_info);

    if (pci_channel_io_perm_failure == error)
    {
        return PCI_ERS_RESULT_DISCONNECT;
    }

    //
    // For NPU devices we need to determine if its FREEZE/FENCE EEH, which
    // requires a register read.
    // Tell Linux to continue recovery of the device. The kernel will enable
    // MMIO for the NPU and call the mmio_enabled callback.
    //
    return PCI_ERS_RESULT_CAN_RECOVER;
}

static pci_ers_result_t ibmnpu_pci_mmio_enabled
(
    struct pci_dev *dev
)
{
    if (NULL == dev)
    {
        return PCI_ERS_RESULT_NONE;
    }

    nvlink_print(NVLINK_DBG_ERRORS,
        "IBMNPU: ibmnpu_pci_mmio_enabled device " NV_PCI_DEV_FMT "\n",
        NV_PCI_DEV_FMT_ARGS(dev));

    //
    // It is understood that we will not attempt to recover from an EEH, but 
    // IBM has requested that we indicate in the logs that it occured and
    // that it was either a FREEZE or a FENCE.
    //
    // Within the MMIO handler specifically, a persistent failure condition
    // is considered a FENCE condition which requires a system power cycle.
    //
    if (ibmnpu_os_device_check_failure(dev))
    {
        nvlink_print(NVLINK_DBG_ERRORS,
            "IBMNPU: NPU FENCE detected, machine power cycle required.\n");
    }
    else
    {
        nvlink_print(NVLINK_DBG_ERRORS,
            "IBMNPU: NPU FREEZE detected, driver reload required.\n");
    }

    nvlink_print(NVLINK_DBG_ERRORS,
        "IBMNPU: Disconnecting device " NV_PCI_DEV_FMT "\n",
        NV_PCI_DEV_FMT_ARGS(dev));

    // There is no way out at this point, request a disconnect.
    return PCI_ERS_RESULT_DISCONNECT;
}

NvBool ibmnpu_os_device_check_failure(void *handle)
{
    NvU16 pci_vendor; 

    //
    // According to IBM, any config cycle read of all Fs will cause the
    // firmware to check for an EEH failure on the associated device.
    // If the EEH failure condition exists, EEH error handling will be
    // triggered and PCIBIOS_DEVICE_NOT_FOUND will be returned.    
    //
    return (pci_read_config_word(handle, PCI_VENDOR_ID, &pci_vendor) == PCIBIOS_DEVICE_NOT_FOUND) ? NV_TRUE : NV_FALSE;
}

int ibmnpu_init(void)
{
    NvlStatus retval = ibmnpu_lib_load(0xFFFFFFFF, 0xFFFFFFFF);

    if (NVL_SUCCESS != retval)
    {
        nvlink_print(NVLINK_DBG_ERRORS,
            "Failed to load ibmnpu library : %d\n", retval);
        return -1;
    }

    return 0;
}

void ibmnpu_exit(void)
{
    NvlStatus retval = ibmnpu_lib_unload();

    if (NVL_SUCCESS != retval)
    {
        nvlink_print(NVLINK_DBG_ERRORS,
            "Error occurred while unloading ibmnpu library : %d\n", retval);
    }
}

NvlStatus ibmnpu_init_device(struct pci_dev *dev)
{
    NvlStatus retval;
    ibmnpu_brick_info_t *brick_info = pci_get_drvdata(dev);

    if (NULL == brick_info)
    {
        return NVL_ERR_INVALID_STATE;
    }

    if (brick_info->registered)
    {
        return NVL_SUCCESS;
    }

    /* Create and register the device in nvlink core library */
    retval = ibmnpu_lib_register_device(NV_PCI_DOMAIN_NUMBER(dev),
                                        NV_PCI_BUS_NUMBER(dev),
                                        NV_PCI_SLOT_NUMBER(dev),
                                        PCI_FUNC(dev->devfn),
                                        dev);

    if (NVL_UNBOUND_DEVICE == retval)
    {
        nvlink_print(NVLINK_DBG_SETUP,
            "IBMNPU: No GPU is associated to this brick, skipping.\n");
        return retval;
    }

    if (NVL_SUCCESS != retval)
    {
        nvlink_print(NVLINK_DBG_ERRORS,
            "IBMNPU: Failed to register NPU device (%d)\n", retval);
        return retval;
    }

    /* Now actually start the device initialization */
    retval = ibmnpu_lib_initialize_device(dev);
    if (NVL_SUCCESS != retval)
    {
        nvlink_print(NVLINK_DBG_ERRORS,
            "Failed to initialize NPU device " NV_PCI_DEV_FMT " : %d\n",
            NV_PCI_DEV_FMT_ARGS(dev), retval);
        ibmnpu_lib_unregister_device(dev);
        return retval;
    }

    brick_info->registered = NV_TRUE;

    return NVL_SUCCESS;
}

void ibmnpu_unregister_device(struct pci_dev *dev)
{
    ibmnpu_brick_info_t *brick_info = pci_get_drvdata(dev);

    ibmnpu_lib_unregister_device(dev);
    brick_info->registered = NV_FALSE;
}

NvlStatus ibmnpu_lib_load
(
    NvU32 accepted_domain,
    NvU32 accepted_link_mask
)
{
    NvlStatus retval = NVL_SUCCESS;
    int rc;

    ibmnpu_init_watchdog();

    retval = ibmnpu_lib_initialize(accepted_domain, accepted_link_mask);
    if (NVL_SUCCESS != retval)
    {
        nvlink_print(NVLINK_DBG_ERRORS,
            "Failed to initialize ibmnpu driver : %d\n", retval);

        goto ibmnpu_lib_initialize_fail;
    }

    rc = pci_register_driver(&ibmnpu_pci_driver);
    if (rc < 0)
    {
        nvlink_print(NVLINK_DBG_ERRORS, 
            "Failed to register ibmnpu driver : %d\n", rc);

        retval = (NvlStatus)rc;
        goto pci_register_driver_fail;
    }

    return retval;

pci_register_driver_fail:
    ibmnpu_lib_shutdown();

ibmnpu_lib_initialize_fail:
    ibmnpu_shutdown_watchdog();

    return retval;
}

NvlStatus ibmnpu_os_device_enable_pci(void *handle)
{
    if (NULL == handle)
    {
        return -NVL_BAD_ARGS;
    }

    /* PCI enablement happens as a part of ibmnpu_probe() on Linux */
    return NVL_SUCCESS;
}

NvlStatus ibmnpu_os_device_load_bar_info(void *handle, nvlink_pci_info *info)
{
    struct pci_dev *dev = handle;
    ibmnpu_brick_info_t *brick_info;

    if (NULL == handle || NULL == info)
    {
        return -NVL_BAD_ARGS;
    }

    brick_info = pci_get_drvdata(dev);
    if (NULL == brick_info)
    {
        return -NVL_ERR_INVALID_STATE;
    }

    /*
     * Copy our info/mappings from the info that was created during
     * ibmnpu_probe.
     */
    memcpy(info->bars, brick_info->pci_info.bars, sizeof(info->bars));

    return NVL_SUCCESS;
}

NvlStatus ibmnpu_lib_unload(void)
{
    NvlStatus retval;

    retval = ibmnpu_lib_shutdown();
    if (NVL_SUCCESS != retval)
    {
        nvlink_print(NVLINK_DBG_ERRORS,
            "Failed to shutdown ibmnpu driver : %d\n", retval);
    }

    pci_unregister_driver(&ibmnpu_pci_driver);

    ibmnpu_shutdown_watchdog();

    return retval;
}

NvlStatus ibmnpu_os_device_enable_irq(void *handle, nvlink_pci_info *info)
{
    struct pci_dev *dev = handle;
    int rc;

    if (NULL == handle || NULL == info)
    {
        return -NVL_BAD_ARGS;
    }

    if (info->intHooked)
    {
        nvlink_print(NVLINK_DBG_SETUP,
            "ibmnpu interrupt already initialized\n");
        return NVL_SUCCESS;
    }

    info->irq = dev->irq;

    rc = request_irq(info->irq, ibmnpu_isr, IRQF_SHARED, IBMNPU_DEVICE_NAME,
                     (void *)info);
    if (rc != 0)
    {
        nvlink_print(NVLINK_DBG_ERRORS,
            "NPU device failed to get irq (%d)\n", rc);
        return -NVL_PCI_ERROR;
    }

    info->intHooked = NV_TRUE;

    return NVL_SUCCESS;
}

NvlStatus ibmnpu_os_device_disable_irq(void *handle, nvlink_pci_info *info)
{
    if (NULL == handle || NULL == info)
    {
        return -NVL_BAD_ARGS;
    }

    if (!info->intHooked)
    {
        nvlink_print(NVLINK_DBG_SETUP, "ibmnpu interrupt not wired up\n");
        return NVL_SUCCESS;
    }

    free_irq(info->irq, (void *)info);

    info->intHooked = NV_FALSE;

    return NVL_SUCCESS;
}

NvlStatus ibmnpu_os_device_unload_bar_info(void *handle, nvlink_pci_info *info)
{
    NvlStatus retval = NVL_SUCCESS;
    unsigned int bar;

    if (NULL == handle || NULL == info)
    {
        return -NVL_BAD_ARGS;
    }

    if (NULL == info->bars[0].pBar)
    {
        nvlink_print(NVLINK_DBG_WARNINGS,
            "Cannot unmap ibmnpu device bars: not initialized.\n");
        return retval;
    }

    for (bar = 0; bar < IBMNPU_MAX_BARS; bar++)
    {
        if (NULL != info->bars[bar].pBar)
        {
            /* The actual unmapping happens during ibmnpu_remove */
            info->bars[bar].pBar = NULL;
        }
    }

    return retval;
}

NvlStatus ibmnpu_os_device_disable_pci(void *handle)
{
    if (NULL == handle)
    {
        return -NVL_BAD_ARGS;
    }

    /* PCI disablement happens as a part of ibmnpu_remove() on Linux */
    return NVL_SUCCESS;
}

NvlStatus ibmnpu_os_device_release(void *handle)
{
    return NVL_SUCCESS;
}

NvU8 ibmnpu_os_device_pci_read_08 (void *handle, NvU32 offset)
{
    NvU8 buffer = 0xFF;
    if (NULL == handle || offset > NV_PCIE_CFG_MAX_OFFSET)
    {
        return buffer;
    }

    pci_read_config_byte(handle, offset, &buffer);
    return buffer;
}

NvU16 ibmnpu_os_device_pci_read_16 (void *handle, NvU32 offset)
{
    NvU16 buffer = 0xFFFF;
    if (NULL == handle || offset > NV_PCIE_CFG_MAX_OFFSET)
    {
        return buffer;
    }

    pci_read_config_word(handle, offset, &buffer);
    return buffer;
}

NvU32 ibmnpu_os_device_pci_read_32 (void *handle, NvU32 offset)
{
    NvU32 buffer = 0xFFFFFFFF;
    if (NULL == handle || offset > NV_PCIE_CFG_MAX_OFFSET)
    {
        return buffer;
    }

    pci_read_config_dword(handle, offset, &buffer);
    return buffer;
}

void ibmnpu_os_device_pci_write_08(void *handle, NvU32 offset, NvU8  data)
{
    if (NULL == handle || offset > NV_PCIE_CFG_MAX_OFFSET)
    {
        return;
    }
    
    pci_write_config_byte(handle, offset, data);
}

void ibmnpu_os_device_pci_write_16(void *handle, NvU32 offset, NvU16 data)
{
    if (NULL == handle || offset > NV_PCIE_CFG_MAX_OFFSET)
    {
        return;
    }
    
    pci_write_config_word(handle, offset, data);
}

void ibmnpu_os_device_pci_write_32(void *handle, NvU32 offset, NvU32 data)
{
    if (NULL == handle || offset > NV_PCIE_CFG_MAX_OFFSET)
    {
        return;
    }
    
     pci_write_config_dword(handle, offset, data);
}

void
ibmnpu_device_get_genregs_info
(
    struct pci_dev *npu_dev,
    ibmnpu_genregs_info_t *genregs_info
)
{
    ibmnpu_brick_info_t *brick_info = NULL;
    nvlink_pci_info *pci_info = NULL;

    genregs_info->start_addr = IBMNPU_INVALID_PHYS_ADDR;
    genregs_info->start_ptr = NULL;
    genregs_info->size = 0;

    /*
     * The generation registers needed for relaxed ordering mode
     * synchronization are located in the NPU device BAR1, if present.
     * It's up to the caller to select the right NPU struct pci_dev (for
     * the set of generation registers needed).
     */
    if ((NULL == npu_dev) ||
        (NULL == (brick_info = pci_get_drvdata(npu_dev))) ||
        (0 == brick_info->pci_info.bars[1].barSize))
    {
        return;
    }

    pci_info = &brick_info->pci_info;

    /*
     * We expect callers will only use the subranges of the generation
     * registers that correspond to 8-byte little-endian format. However,
     * because this region is mapped by the Resource Manager and all
     * generation registers are within a single 64K page, we let them handle
     * the layout and just provide the entire BAR. If this region layout ends
     * up changing at some point in future hardware, we may need to provide a
     * layout cue/version as well.
     */
    WARN_ON(pci_info->bars[1].barSize != PAGE_SIZE);
    genregs_info->start_addr = pci_info->bars[1].baseAddr;
    genregs_info->start_ptr = pci_info->bars[1].pBar;
    genregs_info->size = NvU64_LO32(pci_info->bars[1].barSize);
}

#define NPU_OF_PROPERTY_VAL_TO_NvU64(v, idx) \
    ((((NvU64)be32_to_cpu(v[idx])) << 32) | be32_to_cpu(v[idx+1]))

static NvU64
ibmnpu_device_get_target_addr
(
    struct device_node *npu_node
)
{
    NvU64 spa = IBMNPU_INVALID_PHYS_ADDR;
#if defined(NV_OF_GET_PROPERTY_PRESENT)
    int len;
    const NvU32 *val;
    
    val = of_get_property(npu_node, "ibm,device-tgt-addr", &len);
    if (!val || len != sizeof(NvU64))
    {
        return IBMNPU_INVALID_PHYS_ADDR;
    }

    spa = NPU_OF_PROPERTY_VAL_TO_NvU64(val, 0);
#endif
    return spa;
}

static int
ibmnpu_device_get_memory_node_info
(
    struct pci_dev *npu_dev,
    struct device_node *npu_node,
    NvU64 *base_addr,
    NvU64 *size
)
{
    int nid = NUMA_NO_NODE;
#if defined(NV_OF_GET_PROPERTY_PRESENT) && \
    defined(NV_OF_FIND_NODE_BY_PHANDLE_PRESENT) && \
    defined(NV_OF_NODE_TO_NID_PRESENT) && \
    !NV_IS_EXPORT_SYMBOL_GPL_of_node_to_nid
    const NvU32 *val;
    const NvU32 *mem_phandle;
    struct device_node *mem_node;
    int len;

    mem_phandle = of_get_property(npu_node, "memory-region", NULL);
    if (!mem_phandle)
    {
        return NUMA_NO_NODE;
    }

    mem_node = of_find_node_by_phandle(be32_to_cpu(*mem_phandle));
    if (!mem_node)
    {
        nvlink_print(NVLINK_DBG_SETUP,
            "IBMNPU: no memory node found for NPU device " NV_PCI_DEV_FMT "\n",
            NV_PCI_DEV_FMT_ARGS(npu_dev));
        return NUMA_NO_NODE;
    }

    nid = of_node_to_nid(mem_node);
    if (nid == NUMA_NO_NODE)
    {
        goto release_mem_node;
    }

    val = (NvU32 *)of_get_property(mem_node, "reg", &len);
    if (!val || len != 2 * sizeof(NvU64))
    {
        nvlink_print(NVLINK_DBG_ERRORS,
            "IBMNPU: no valid 'reg' property found on memory node for NPU device"
            " " NV_PCI_DEV_FMT "\n", NV_PCI_DEV_FMT_ARGS(npu_dev));
        nid = NUMA_NO_NODE;
        goto release_mem_node;
    }

    *base_addr = NPU_OF_PROPERTY_VAL_TO_NvU64(val, 0);
    *size = NPU_OF_PROPERTY_VAL_TO_NvU64(val, 2);

release_mem_node:
    of_node_put(mem_node);
#endif
    return nid;
}

int
ibmnpu_device_get_memory_config
(
    struct pci_dev *npu_dev,
    NvU64 *device_tgt_addr,
    NvU64 *base_addr,
    NvU64 *size
)
{
    struct device_node *npu_node = pci_device_to_OF_node(npu_dev);

    if (!npu_node)
    {
        nvlink_print(NVLINK_DBG_SETUP,
            "IBMNPU: no OF node found for NPU device " NV_PCI_DEV_FMT "\n",
            NV_PCI_DEV_FMT_ARGS(npu_dev));
        return NUMA_NO_NODE;
    }

    *device_tgt_addr = ibmnpu_device_get_target_addr(npu_node);
    if (*device_tgt_addr == IBMNPU_INVALID_PHYS_ADDR)
    {
        nvlink_print(NVLINK_DBG_SETUP,
            "IBMNPU: no device target address found for NPU device "
            NV_PCI_DEV_FMT "\n", NV_PCI_DEV_FMT_ARGS(npu_dev));
        return NUMA_NO_NODE;
    }

    return ibmnpu_device_get_memory_node_info(npu_dev, npu_node,
                                              base_addr, size);
}

int
ibmnpu_device_get_chip_id
(
    struct pci_dev *npu_dev
)
{
#if defined(NV_OF_GET_IBM_CHIP_ID_PRESENT)
    struct device_node *npu_node = pci_device_to_OF_node(npu_dev);

    if (!npu_node)
    {
        nvlink_print(NVLINK_DBG_SETUP,
            "IBMNPU: no OF node found for NPU device " NV_PCI_DEV_FMT "\n",
            NV_PCI_DEV_FMT_ARGS(npu_dev));
        return -1;
    }

    return of_get_ibm_chip_id(npu_node);
#else
    return -1;
#endif
}
