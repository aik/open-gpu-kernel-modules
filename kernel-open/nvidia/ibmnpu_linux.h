/*******************************************************************************
    Copyright (c) 2017-2019 NVIDIA Corporation

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

#ifndef _IBMNPU_LINUX_H_
#define _IBMNPU_LINUX_H_

#include <linux/types.h>

#include "nvlink_common.h"

int       ibmnpu_init             (void);
void      ibmnpu_exit             (void);
NvlStatus ibmnpu_init_device      (struct pci_dev *);
void      ibmnpu_unregister_device(struct pci_dev *);

typedef struct ibmnpu_brick_info_s {
    struct pci_dev  *dev;
    nvlink_pci_info  pci_info;
    NvBool           registered;
} ibmnpu_brick_info_t;

#define IBMNPU_INVALID_PHYS_ADDR    (NV_U64_MAX)

typedef struct ibmnpu_genregs_info_s {
    NvU64 start_addr;
    void *start_ptr;
    NvU32 size;
} ibmnpu_genregs_info_t;

void ibmnpu_device_get_genregs_info(struct pci_dev *npu_dev,
                                    ibmnpu_genregs_info_t *genregs_info);

int ibmnpu_device_get_memory_config(struct pci_dev *npu_dev,
                                    NvU64 *device_tgt_addr,
                                    NvU64 *base_addr,
                                    NvU64 *size);

int ibmnpu_device_get_chip_id(struct pci_dev *npu_dev);
struct pci_dev *pnv_pci_get_gpu_dev(struct pci_dev *npdev);
struct pci_dev *pnv_pci_get_npu_dev(struct pci_dev *gpdev, int index);


#endif
