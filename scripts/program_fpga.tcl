# PQC Scheduler - FPGA Programming Script for Vivado
# Copyright (c) 2025 Dyber, Inc. All Rights Reserved.
#
# Usage: vivado -mode batch -source scripts/program_fpga.tcl \
#              -tclargs <bitstream_file> [device_index]
#
# Example: vivado -mode batch -source scripts/program_fpga.tcl \
#                -tclargs bitstreams/mlkem768_ntt_250mhz.bit 0

puts "========================================"
puts "PQC Scheduler FPGA Programmer"
puts "========================================"
puts ""

# Parse arguments
if {$argc < 1} {
    puts "Usage: vivado -mode batch -source program_fpga.tcl -tclargs <bitstream> \[device_index\]"
    puts ""
    puts "  bitstream:    Path to .bit file"
    puts "  device_index: Hardware device index (default: 0)"
    puts ""
    puts "Available bitstreams:"
    foreach f [glob -nocomplain bitstreams/*.bit] {
        puts "  $f"
    }
    exit 1
}

set BITSTREAM [lindex $argv 0]
set DEVICE_INDEX 0
if {$argc > 1} {
    set DEVICE_INDEX [lindex $argv 1]
}

# Validate bitstream file
if {![file exists $BITSTREAM]} {
    puts "ERROR: Bitstream file not found: $BITSTREAM"
    exit 1
}

puts "Bitstream:    $BITSTREAM"
puts "Device Index: $DEVICE_INDEX"
puts ""

# Open hardware manager
puts "Opening hardware manager..."
open_hw_manager

# Connect to hardware server
puts "Connecting to hardware server..."
if {[catch {connect_hw_server -allow_non_jtag} result]} {
    puts "Could not connect to local hw_server, trying to start one..."
    if {[catch {exec hw_server &} result]} {
        puts "ERROR: Could not start hw_server: $result"
        exit 1
    }
    after 2000
    connect_hw_server -allow_non_jtag
}

# Find hardware targets
puts ""
puts "Scanning for hardware targets..."
set targets [get_hw_targets]

if {[llength $targets] == 0} {
    puts "ERROR: No hardware targets found"
    puts "Please ensure:"
    puts "  1. FPGA board is connected via JTAG/USB"
    puts "  2. Board is powered on"
    puts "  3. JTAG drivers are installed"
    close_hw_manager
    exit 1
}

puts "Found [llength $targets] target(s):"
set idx 0
foreach target $targets {
    puts "  \[$idx\] $target"
    incr idx
}
puts ""

# Select target
if {$DEVICE_INDEX >= [llength $targets]} {
    puts "ERROR: Device index $DEVICE_INDEX out of range (0-[expr {[llength $targets]-1}])"
    close_hw_manager
    exit 1
}

set target [lindex $targets $DEVICE_INDEX]
puts "Selecting target: $target"
open_hw_target $target

# Find FPGA devices on target
set devices [get_hw_devices]

if {[llength $devices] == 0} {
    puts "ERROR: No FPGA devices found on target"
    close_hw_target
    close_hw_manager
    exit 1
}

puts "Found [llength $devices] device(s):"
foreach device $devices {
    set part [get_property PART $device]
    puts "  $device ($part)"
}
puts ""

# Select first FPGA device
set device [lindex $devices 0]
puts "Selecting device: $device"
current_hw_device $device

# Set programming file
puts "Setting bitstream: $BITSTREAM"
set_property PROGRAM.FILE $BITSTREAM $device

# Check if probes file exists
set ltx_file [file rootname $BITSTREAM].ltx
if {[file exists $ltx_file]} {
    puts "Debug probes file found: $ltx_file"
    set_property PROBES.FILE $ltx_file $device
}

# Program the device
puts ""
puts "========================================"
puts "Programming FPGA..."
puts "========================================"

if {[catch {program_hw_devices $device} result]} {
    puts ""
    puts "ERROR: Programming failed: $result"
    close_hw_target
    close_hw_manager
    exit 1
}

puts ""
puts "Programming successful!"

# Verify device
puts ""
puts "Verifying device status..."

refresh_hw_device $device

set status [get_property REGISTER.CONFIG_STATUS $device]
puts "Config Status: $status"

set done [get_property REGISTER.BOOT_STATUS $device]
puts "Boot Status:   $done"

# Check for ILA cores
puts ""
puts "Checking for debug cores..."
set ila_cores [get_hw_ilas -quiet]
if {[llength $ila_cores] > 0} {
    puts "Found [llength $ila_cores] ILA core(s):"
    foreach ila $ila_cores {
        puts "  $ila"
    }
} else {
    puts "No ILA debug cores found"
}

# Close connections
puts ""
puts "Closing connections..."
close_hw_target
close_hw_manager

puts ""
puts "========================================"
puts "FPGA Programming Complete"
puts "========================================"
puts ""
puts "The FPGA is now programmed with: $BITSTREAM"
puts ""
puts "Next steps:"
puts "  1. Verify PCIe device is detected: lspci | grep Xilinx"
puts "  2. Load XDMA driver: sudo modprobe xdma"
puts "  3. Check device node: ls /dev/xdma*"
puts "  4. Run scheduler with FPGA: cargo run --release --features fpga"
puts ""

exit 0
