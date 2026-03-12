import io
from environment_monitor import load_environment_data, run_monitor
from energy_prediction import load_energy_data, run_prediction
from predictive_maintenance import load_sensor_data, run_maintenance

print('Env test')
csv='timestamp,temperature,humidity,air_quality\n2026-01-01 00:00,20,30,50'
df=load_environment_data(io.StringIO(csv))
print(run_monitor(df))

print('Energy test')
csv2='timestamp,energy_usage\n2026-01-01 00:00,100'
df2=load_energy_data(io.StringIO(csv2))
print(run_prediction(df2))

print('Maintenance test')
csv3='timestamp,vibration,temperature,failure\n2026-01-01 00:00,0.1,35,0\n2026-01-01 01:00,0.2,40,1'
df3=load_sensor_data(io.StringIO(csv3))
print(run_maintenance(df3))
