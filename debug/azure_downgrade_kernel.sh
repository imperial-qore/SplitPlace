sudo apt remove -y -f linux-headers-$(uname -r) \
linux-image-$(uname -r) \
linux-image-unsigned-$(uname -r)

sudo apt install -y -f linux-image-4.15.0-1009-azure \
linux-tools-4.15.0-1009-azure \
linux-cloud-tools-4.15.0-1009-azure \
linux-headers-4.15.0-1009-azure \
linux-modules-4.15.0-1009-azure \
linux-modules-extra-4.15.0-1009-azure

sudo reboot