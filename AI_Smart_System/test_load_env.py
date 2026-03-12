import environment_monitor as em

df = em.load_environment_data('datasets/environment_data.csv')
print(df.head())
