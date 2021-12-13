#!/bin/bash

trap "clean_exit" SIGINT
clean_exit() {
  echo -e "\nStopping capture and exiting safely\n"
  sudo pkill -2 tcpdump
  exit 1
}

echo -n "Enter the IP address of the device [192.168.1.2]: "
read -p "" ip_addr
ip_addr=${ip_addr:-"192.168.1.2"}
echo -e "Using address $ip_addr\n"

echo -n "Enter the interface to listen on [wlan0]: "
read -p "" interface
interface=${interface:-"wlan0"}
echo -e "Capturing traffic from $interface\n"

sudo rm ./predict/* 2> /dev/null
sudo tcpdump -U -i $interface -w ./predict/demo.pcap "host $ip_addr" &

echo -e "Press any key to stop capturing"
while [ true ]; do
        read -n 1
if [ $? = 0 ]; then
        sudo pkill -2 tcpdump
        break
fi
done
