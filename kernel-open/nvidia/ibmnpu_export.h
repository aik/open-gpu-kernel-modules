/*******************************************************************************
    Copyright (c) 2015-2019 NVidia Corporation

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


#ifndef _IBMNPU_EXPORT_H_
#define _IBMNPU_EXPORT_H_

#ifdef __cplusplus
extern "C" {
#endif

#include "nvlink_common.h"

#define IBMNPU_DRIVER_NAME                              "ibmnpu"
#define IBMNPU_DEVICE_NAME                              "NPU Device"
#define IBMNPU_LINK_NAME                                "NPU Link"
#define IBMNPU_MAX_BARS                                 2

#define PCI_CLASS_BRIDGE_NPU                            0x0680

#define PCI_DEVICE_ID_IBM_NPU                           0x04ea

#define PCI_VENDOR_ID_IBM                               0x1014

#define PCI_REVISION_ID_IBM_NPU_P8                      0x0
#define PCI_REVISION_ID_IBM_NPU_P9                      0x1
#define PCI_REVISION_ID_IBM_NPU_P9P                     0x2

/*
 * @Brief : Initializes and registers the NPU driver with NVlink.
 *
 * @Description :
 *
 * @param[in] accepted_domain   - Accepted NPU domain. Links which
 *                                  appear on domains other than this number
 *                                  will be ignored, unless the accepted domain
 *                                  is 0xFFFFFFFF.
 * @param[in] accepted_link_mask - Mask of accepted links. Link indices whose bits
 *                                  are not raised in this mask will be ignored.
 *
 * @returns                 NVL_SUCCESS if action succeeded,
 *                          an NVL error code otherwise
 */
NvlStatus ibmnpu_lib_initialize
(
    NvU32 accepted_domain,
    NvU32 accepted_link_mask
);

/*
 * @Brief : Shuts down and unregisters the driver/devices from the NVlink
 *              library.
 *
 * @Description :
 *
 * @returns                 NVL_SUCCESS if the action succeeded
 *                          an NVL error code otherwise
 */
NvlStatus ibmnpu_lib_shutdown(void);

/*
 * @Brief : Creates and registers a device with the given data with the nvlink
 *              core library.
 *
 * @Description :
 *
 * @param[in] domain        pci domain of the device
 * @param[in] bus           pci bus of the device
 * @param[in] device        pci device of the device
 * @param[in] func          pci function of the device
 * @param[in] handle        Device handle used to interact with OS layer
 *
 * @returns                 NVL_SUCCESS if the action succeeded
 *                          an NVL error code otherwise
 */
NvlStatus ibmnpu_lib_register_device
(
    NvU16 domain, 
    NvU8 bus, 
    NvU8 device, 
    NvU8 func,
    void *handle
);

/*
 * @Brief : Unregisters and destroys the device with the nvlink core library.
 *
 * @Description :
 *
 * @param[in] handle        Device handle used to interact with OS layer
 *
 * @returns                 void
 */
void ibmnpu_lib_unregister_device(void *handle);

/*
 * @Brief : Initializes a device associated with the given handle.
 *
 * @Description :
 *
 * @param[in] handle        Device handle used to interact with OS layer
 *
 * @returns                 NVL_SUCCESS if device is successfully initialized
 *                          an NVL error code otherwise
 */
NvlStatus ibmnpu_lib_initialize_device(void *handle);

/*
 * @Brief : Services an interrupt triggered for the device with the given info.
 *
 * @Description :
 *
 * @param[in] info          reference to the pci info of the device to service
 *
 * @returns                 NVL_SUCCESS if the action succeeded
 *                          -NVL_BAD_ARGS if bad arguments provided
 */
NvlStatus ibmnpu_lib_service_device(nvlink_pci_info *info);

/*
 * @Brief : Notifies the core to avoid MMIO for the device with the given info.
 *
 * @Description :
 *
 * @param[in] info          reference to the pci info of the device to abort
 *
 * @returns                 void
 */
void ibmnpu_lib_stop_device_mmio(nvlink_pci_info *info);

/*
 * Initializes ibmnpu library, preparing the driver to register
 *     discovered devices into the core library.
 */
NvlStatus ibmnpu_lib_load
(
    NvU32 accepted_domain,
    NvU32 accepted_link_mask
);

/*
 * Shuts down the ibmnpu library, deregistering its devices from
 *     the core and freeing core operating system accounting info.
 */
NvlStatus ibmnpu_lib_unload(void);

/*
 * Initializes the pci bus for the given device, including
 *     enabling device memory transactions and bus mastering.
 */
NvlStatus ibmnpu_os_device_enable_pci(void *handle);

/*
 * Maps the device base address registers into CPU memory, and
 *     populates the device pci data with the mapping.
 */
NvlStatus ibmnpu_os_device_load_bar_info(void *handle, nvlink_pci_info *info);

/*
 * Registers an interrupt service routine with the operating system
 *     to handle device interrupts.
 */
NvlStatus ibmnpu_os_device_enable_irq
(
    void *handle,
    nvlink_pci_info *info
);

/*
 * Disables the pci bus for the given device.
 */
NvlStatus ibmnpu_os_device_disable_pci(void *handle);

/*
 * Unmaps the previously mapped base address registers from cpu memory.
 */
NvlStatus ibmnpu_os_device_unload_bar_info(void *handle, nvlink_pci_info *info);

/*
 * Unregisters the interrupt service routine from the operating system.
 */
NvlStatus ibmnpu_os_device_disable_irq
(
    void *handle,
    nvlink_pci_info *info
);

/*
 * Cleans up any state the OS layer allocated for this device.
 */
NvlStatus ibmnpu_os_device_release(void *handle);

/*
 * Detects failure condition on the requested device.
 */
NvBool ibmnpu_os_device_check_failure(void *handle);

NvU8  ibmnpu_os_device_pci_read_08 (void *handle, NvU32 offset);
NvU16 ibmnpu_os_device_pci_read_16 (void *handle, NvU32 offset);
NvU32 ibmnpu_os_device_pci_read_32 (void *handle, NvU32 offset);
void  ibmnpu_os_device_pci_write_08(void *handle, NvU32 offset, NvU8  data);
void  ibmnpu_os_device_pci_write_16(void *handle, NvU32 offset, NvU16 data);
void  ibmnpu_os_device_pci_write_32(void *handle, NvU32 offset, NvU32 data);

#ifdef __cplusplus
}
#endif

#endif //_IBMNPU_EXPORT_H_
