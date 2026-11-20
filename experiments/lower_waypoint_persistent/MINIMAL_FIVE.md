# Minimal learning diagnostic

1 UAV, 5 fixed sensors, 5 physical steps = 5 actor decisions per episode.
No service-group rotation, energy, charging or return; reset position and queues
every episode. Map 600m square; start (300,300); sensors on radius-140m pentagon;
coverage radius 40m. Speed remains 6m/s, max movement time 30s (180m).
A clockwise sensor-center tour reaches all five in five steps: first 140m,
then four edges of about 164.6m. Overlapping coverage cannot clear two sensors.
Same actor/critic attention, GELU, status feature, reward and SAC settings as v5.
Gamma .95. Train uses actor distribution sampling only (epsilon=0); eval uses
tanh(mu). Initial budget 20000 steps = 4000 episodes; eval/checkpoint every 1000.
This diagnoses a smaller task; successful training does not prove full-map success.
