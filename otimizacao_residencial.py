from ortools.sat.python import cp_model
import pandas as pd
import matplotlib.pyplot as plt


def otimizar_agendamento_residencial():
    dados_aparelhos = [
        (1, 'Carro Elétrico', 7000, 4, 22, 6),
        (1, 'Boiler', 2000, 2, 0, 5),
        (2, 'Carro Elétrico', 7000, 5, 21, 7),
        (2, 'Lava-louças', 1500, 1, 23, 4),
        (3, 'Boiler', 2000, 3, 22, 6),
        (3, 'Piscina', 1000, 2, 1, 6),
        (4, 'Carro Elétrico', 7000, 3, 23, 6),
        (5, 'Boiler', 2000, 2, 0, 5),
        (6, 'Carro Elétrico', 7000, 6, 20, 6),
        (6, 'Lava-louças', 1500, 1, 22, 3),
        (7, 'Boiler', 2000, 2, 23, 5),
        (8, 'Carro Elétrico', 7000, 4, 22, 7),
        (9, 'Piscina', 1000, 3, 0, 6),
        (10, 'Carro Elétrico', 7000, 5, 21, 6),
        (10, 'Boiler', 2000, 2, 0, 4),
    ]

    horizonte = 24 * 60
    modelo = cp_model.CpModel()
    variaveis = {}

    for res_id, aparelho, potencia, dur_h, ini_h, fim_h in dados_aparelhos:
        tarefa_id = f'Res{res_id}_{aparelho}'
        dur_min = int(dur_h * 60)
        ini_min = ini_h * 60
        fim_min = fim_h * 60

        if fim_min <= ini_min:
            fim_min += 1440

        inicio_var = modelo.NewIntVar(ini_min, fim_min - dur_min, f'{tarefa_id}_inicio')
        intervalo_var = modelo.NewIntervalVar(inicio_var, dur_min, inicio_var + dur_min, f'{tarefa_id}_intervalo')

        variaveis[tarefa_id] = {
            'intervalo': intervalo_var,
            'potencia': potencia,
            'inicio': inicio_var,
            'duracao': dur_min
        }

    demanda_por_minuto = [modelo.NewIntVar(0, 100000, f'demanda_{m}') for m in range(horizonte)]

    for m in range(horizonte):
        contrib = []
        for tarefa_id, var in variaveis.items():
            start = var['inicio']
            dur = var['duracao']
            potencia = var['potencia']

            ativo = modelo.NewBoolVar(f'{tarefa_id}_ativo_em_{m}')
            modelo.Add(m >= start).OnlyEnforceIf(ativo)
            modelo.Add(m < start + dur).OnlyEnforceIf(ativo)
            contrib.append(ativo * potencia)

        modelo.Add(demanda_por_minuto[m] == sum(contrib))

    pico = modelo.NewIntVar(0, 100000, 'pico_demanda')
    modelo.AddMaxEquality(pico, demanda_por_minuto)
    modelo.Minimize(pico)

    solver = cp_model.CpSolver()
    status = solver.Solve(modelo)

    if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        print("Solução encontrada!")
        print(f"Pico de Demanda Otimizado: {solver.ObjectiveValue() / 1000:.2f} kW")

        agendamento = []
        for tarefa_id, var in variaveis.items():
            inicio = solver.Value(var['inicio'])
            agendamento.append({
                'Tarefa': tarefa_id,
                'Início (h)': int((inicio // 60) % 24),
                'Início (min)': int(inicio % 60),
                'Duração (h)': round(var['duracao'] / 60, 2)
            })

        df = pd.DataFrame(agendamento)
        print("\nAgendamento Otimizado:")
        print(df)
        return df, solver.ObjectiveValue(), dados_aparelhos
    else:
        print("Não foi possível encontrar uma solução.")
        return None, None, None


def calcular_demanda_nao_otimizada(dados_aparelhos):
    demanda = [0] * 1440

    for _, _, potencia, dur_h, ini_h, fim_h in dados_aparelhos:
        inicio = ini_h * 60
        duracao = int(dur_h * 60)
        fim = fim_h * 60

        if fim <= inicio:
            fim += 1440

        fim_real = min(inicio + duracao, fim)

        for m in range(inicio, fim_real):
            demanda[m % 1440] += potencia

    return demanda


def plotar_resultados(df_agendamento, dados_aparelhos):
    demanda_nao_otimizada = calcular_demanda_nao_otimizada(dados_aparelhos)
    demanda_otimizada = [0] * 1440

    print("Pico da demanda NÃO otimizada:", max(demanda_nao_otimizada) / 1000, "kW")

    if df_agendamento is not None:
        for _, row in df_agendamento.iterrows():
            tarefa = row['Tarefa']
            potencia = next(p for r, a, p, *_ in dados_aparelhos if f"Res{r}_{a}" == tarefa)
            try:
                inicio_min = int(row['Início (h)']) * 60 + int(row['Início (min)'])
                duracao_min = int(float(row['Duração (h)']) * 60)
            except Exception as e:
                print(f"Erro ao processar {tarefa}: {e}")
                continue

            for i in range(duracao_min):
                minuto = (inicio_min + i) % 1440
                demanda_otimizada[minuto] += potencia

    print("Primeiros valores da curva otimizada (W):", demanda_otimizada[:10])

    horas = [i / 60 for i in range(1440)]

    plt.figure(figsize=(15, 7))
    plt.plot(horas, [v / 1000 for v in demanda_nao_otimizada],
             label=f'Demanda Não Otimizada (Pico: {max(demanda_nao_otimizada)/1000:.2f} kW)',
             color='red', linestyle='--')
    plt.plot(horas, [v / 1000 for v in demanda_otimizada],
             label=f'Demanda Otimizada (Pico: {max(demanda_otimizada)/1000:.2f} kW)',
             color='green', linewidth=2)

    plt.title("Comparação: Demanda com e sem Otimização")
    plt.xlabel("Hora do Dia")
    plt.ylabel("Demanda (kW)")
    plt.xticks(range(0, 25, 2))
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("Iniciando otimização de agendamento de cargas residenciais...\n")
    df_agendamento, pico, dados = otimizar_agendamento_residencial()
    if df_agendamento is not None:
        plotar_resultados(df_agendamento, dados)