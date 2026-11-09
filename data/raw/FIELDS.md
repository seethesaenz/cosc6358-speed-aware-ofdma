# Raw pilot output: field definitions (SYNTHETIC data)

One CSV per condition and seed: `<condition>_seed<k>.csv`. One row per channel realization x allocator x rate margin x user.

| Field | Unit | Meaning |
|---|---|---|
| condition | text | perfect_csi, pedestrian_3kmh or vehicle_120kmh |
| seed | integer | NumPy random seed; the same seed gives the same channel draws in every condition |
| realization | integer | Channel drop index within the seed (0 to 199) |
| speed_kmh | km/h | User speed (all users move at this speed in the pilot) |
| tau_ms | ms | Feedback delay between channel measurement and transmission |
| rho | unitless | Channel correlation J0(2*pi*f_D*tau) between estimate and true channel |
| allocator | text | static or sum_rate |
| margin_db | dB | Fixed SNR back-off used when choosing the rate (0 or 3) |
| user | index | User 0 to 3 |
| beta_db | dB | User's average channel gain (0, -6.67, -13.33, -20 dB) |
| n_subcarriers | count | Subcarriers assigned to the user |
| power_w | W | Total power on the user's subcarriers |
| planned_rate_bps | bit/s | Rate chosen from the (possibly outdated) estimate |
| delivered_rate_bps | bit/s | Rate actually delivered on the true channel (0 on subcarriers in outage) |
| n_active_subcarriers | count | User's subcarriers with power > 0 |
| n_outage_subcarriers | count | Active subcarriers whose planned rate exceeded true capacity |
