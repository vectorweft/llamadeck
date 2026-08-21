# OS-level configuration

Settings the kernel resets on every boot, kept here so they are version-controlled
rather than re-typed from shell history.

## `90-r9700-power-cap.rules`

Pins the Radeon AI PRO R9700 to a **220 W** power cap and keeps it out of D3hot
runtime suspend.

Both settings revert on reboot (and the power cap also reverts on driver reload).
The rule matches the card by its **PCI address `0000:03:00.0`**, because `cardN`
and `hwmonN` indices are not stable across boots — on this box the R9700 is
currently `card2`/`hwmon9` and the 9950X iGPU is `card3`/`hwmon10`, but that can
swap. The iGPU sits at `0000:7c:00.0` and does not match.

Install:

    sudo install -m 0644 scripts/os/90-r9700-power-cap.rules /etc/udev/rules.d/
    sudo udevadm control --reload-rules
    sudo udevadm trigger --subsystem-match=hwmon --action=change

Verify (expect `220000000`):

    cat /sys/class/drm/card*/device/hwmon/hwmon*/power1_cap

### Why 220 W

Since 2026-08-20 (user choice): 220 W sits between the two measured
points below — cooler and quieter than 255 W, with prompt throughput expected
between 969 and 893 t/s (~920–940 t/s). The table is the original 2026-08-18
measurement that informed the choice:, Qwen3.8-27B UD-Q6_K_XL on Vulkan1, `pp4096`/`tg128`:

| Cap | pp | tg | Peak temp | Fan |
|---|---|---|---|---|
| 300 W | 1009 t/s | 22.23 | 91–95 °C | 2346 rpm |
| **255 W** | **969 t/s** | **22.09** | **86 °C** | **1786 rpm** |
| 210 W | 893 t/s | 21.85 | 82 °C | 1611 rpm |

The first 45 W costs 4% of prompt throughput and buys 560 rpm of quiet; the next
45 W costs another 7.5% for only 175 rpm more. Generation is bandwidth-bound and
unaffected at any cap. Valid range on this card is 210–300 W.

### Why the runtime-suspend line

As a headless secondary GPU the kernel may suspend the card to D3hot. `amdgpu`
then loses fan control and the fan falls back to a fixed RPM — the configuration
behind reports of 109 °C with a stationary fan. Manual fan control is not possible
on this card (no `pwm1_enable`; ROCm issue #6078), so staying in D0 is the only
guard available.

If you would rather not pin runtime PM, delete the second rule — the power cap
line is independent.
