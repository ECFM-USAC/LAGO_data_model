import matplotlib.pyplot as plt

def plot_readings(df, n=5, legend=True):
    if n == -1:
        data = df
    else:
        data = df.sample(n)

    plt.figure(figsize=(10, 6))

    for _, row in data.iterrows():
        pulse_data = row['readings']
        pulse_data = pulse_data[pulse_data != 0]

        plt.plot(pulse_data, label=row['TBP'])

    plt.xlabel('Índice de lectura')
    plt.ylabel('Valor')
    plt.title(f'Lecturas de {"todos los elementos" if n == -1 else f"{n} elementos aleatorios"}')

    if legend:
        plt.legend(title='TBP')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
