sudo apt install -f linux-image-4.15.0-1009-azure \
linux-tools-4.15.0-1009-azure \
linux-cloud-tools-4.15.0-1009-azure \
linux-headers-4.15.0-1009-azure \
linux-modules-4.15.0-1009-azure \
linux-modules-extra-4.15.0-1009-azure

sudo apt remove -f linux-headers-5.4.0-1026-azure \
linux-image-5.4.0-1026-azure \
linux-image-unsigned-5.4.0-1026-azure

sudo reboot