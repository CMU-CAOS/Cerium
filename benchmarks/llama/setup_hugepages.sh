# 1. Clear out non-essential OS disk caches to free up RAM blocks
echo 3 | sudo tee /proc/sys/vm/drop_caches

# 2. Force the kernel to defragment and compact remaining memory
echo 1 | sudo tee /proc/sys/vm/compact_memory

# 3. Dynamically request 256 GB worth of 2 MB huge pages
echo 131072 | sudo tee /proc/sys/vm/nr_hugepages

#4. Disable memory swapping immediately
sudo sysctl -w vm.swappiness=0

#5. Overcommit memory so the allocation is approved without background evaluation lags
sudo sysctl -w vm.overcommit_memory=1

#6. Tell Transparent Huge Pages to handle defragmentation asynchronously in the background
echo defer | sudo tee /sys/kernel/mm/transparent_hugepage/defrag
