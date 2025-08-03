from ortools.sat.python import cp_model
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def otimizar_agendamento_residencial():
    """
    Otimiza o agendamento de aparelhos residenciais para minimizar o pico de demanda
    """
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

    print("=== DADOS DOS APARELHOS ===")
    print("Residência | Aparelho        | Potência (W) | Duração (h) | Início (h) | Fim (h)")
    print("-" * 85)
    for res_id, aparelho, potencia, dur_h, ini_h, fim_h in dados_aparelhos:
        print(f"    {res_id:2d}     | {aparelho:15s} | {potencia:8d}     | {dur_h:7.1f}     | {ini_h:6d}     | {fim_h:4d}")
    print()

    # Configuração do modelo
    horizonte = 24  # Trabalhar em horas para simplificar
    modelo = cp_model.CpModel()
    
    # Variáveis do modelo
    variaveis = {}
    tarefas_info = {}

    for res_id, aparelho, potencia, dur_h, ini_h, fim_h in dados_aparelhos:
        tarefa_id = f'Res{res_id}_{aparelho.replace(" ", "_")}'
        
        # Ajustar janela de tempo se cruzar meia-noite
        if fim_h <= ini_h:
            fim_h += 24
            
        # Criar variável de início (em horas)
        max_inicio = fim_h - dur_h
        inicio_var = modelo.NewIntVar(ini_h, max_inicio, f'{tarefa_id}_inicio')
        
        # Armazenar informações
        variaveis[tarefa_id] = {
            'inicio': inicio_var,
            'duracao': dur_h,
            'potencia': potencia
        }
        
        tarefas_info[tarefa_id] = {
            'res_id': res_id,
            'aparelho': aparelho,
            'potencia': potencia,
            'duracao': dur_h,
            'janela_inicio': ini_h,
            'janela_fim': fim_h
        }

    # Criar variáveis de demanda por hora
    demanda_por_hora = []
    for h in range(24):
        demanda_por_hora.append(modelo.NewIntVar(0, 100000, f'demanda_hora_{h}'))

    # Modelar a demanda em cada hora
    for h in range(24):
        consumo_nesta_hora = []
        
        for tarefa_id, var_info in variaveis.items():
            inicio_var = var_info['inicio']
            duracao = var_info['duracao']
            potencia = var_info['potencia']
            
            # Criar variáveis booleanas para cada hora de operação
            for dur_offset in range(int(duracao)):
                hora_operacao = h
                
                # Variável booleana: tarefa está ativa nesta hora?
                ativa_var = modelo.NewBoolVar(f'{tarefa_id}_ativa_h{h}_dur{dur_offset}')
                
                # Restrições: ativa somente se a tarefa começou e ainda não terminou
                modelo.Add(inicio_var + dur_offset == hora_operacao).OnlyEnforceIf(ativa_var)
                modelo.Add(inicio_var + dur_offset != hora_operacao).OnlyEnforceIf(ativa_var.Not())
                
                consumo_nesta_hora.append(ativa_var * potencia)
        
        # Definir demanda total nesta hora
        if consumo_nesta_hora:
            modelo.Add(demanda_por_hora[h] == sum(consumo_nesta_hora))
        else:
            modelo.Add(demanda_por_hora[h] == 0)

    # Objetivo: minimizar o pico de demanda
    pico_demanda = modelo.NewIntVar(0, 100000, 'pico_demanda')
    modelo.AddMaxEquality(pico_demanda, demanda_por_hora)
    modelo.Minimize(pico_demanda)

    # Resolver o modelo
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 30.0
    status = solver.Solve(modelo)

    if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        print("✅ Solução encontrada!")
        pico_otimizado = solver.ObjectiveValue()
        print(f"🎯 Pico de Demanda Otimizado: {pico_otimizado / 1000:.2f} kW")
        
        # Extrair agendamento
        agendamento = []
        agendamento_detalhado = {}
        
        for tarefa_id, var_info in variaveis.items():
            inicio_otimizado = solver.Value(var_info['inicio'])
            info = tarefas_info[tarefa_id]
            
            agendamento.append({
                'Tarefa': tarefa_id,
                'Residência': info['res_id'],
                'Aparelho': info['aparelho'],
                'Potência (W)': info['potencia'],
                'Duração (h)': info['duracao'],
                'Início Original (h)': info['janela_inicio'],
                'Início Otimizado (h)': inicio_otimizado % 24,
                'Fim Otimizado (h)': (inicio_otimizado + info['duracao']) % 24
            })
            
            agendamento_detalhado[tarefa_id] = {
                'inicio': inicio_otimizado,
                'duracao': info['duracao'],
                'potencia': info['potencia']
            }

        df = pd.DataFrame(agendamento)
        print("\n=== AGENDAMENTO OTIMIZADO ===")
        print(df.to_string(index=False))
        
        # Calcular demanda otimizada hora por hora
        demanda_otimizada_calculada = [0] * 24
        for tarefa_id, dados in agendamento_detalhado.items():
            inicio = dados['inicio']
            duracao = int(dados['duracao'])
            potencia = dados['potencia']
            
            for h in range(duracao):
                hora_atual = (inicio + h) % 24
                demanda_otimizada_calculada[hora_atual] += potencia
        
        print(f"\n📊 Demanda otimizada calculada (kW por hora): {[d/1000 for d in demanda_otimizada_calculada]}")
        print(f"🔥 Pico calculado: {max(demanda_otimizada_calculada)/1000:.2f} kW")
        
        return df, pico_otimizado, dados_aparelhos, agendamento_detalhado
    else:
        print("❌ Não foi possível encontrar uma solução.")
        return None, None, None, None


def calcular_demanda_nao_otimizada(dados_aparelhos):
    """
    Calcula a demanda sem otimização (horários originais)
    """
    demanda = [0] * 24  # 24 horas

    print("\n=== CÁLCULO DEMANDA NÃO OTIMIZADA ===")
    for res_id, aparelho, potencia, dur_h, ini_h, fim_h in dados_aparelhos:
        print(f"Res{res_id} {aparelho}: {ini_h}h-{ini_h + dur_h}h, {potencia}W")
        
        for h in range(int(dur_h)):
            hora_atual = (ini_h + h) % 24
            demanda[hora_atual] += potencia

    return demanda


def plotar_resultados(df_agendamento, dados_aparelhos, agendamento_detalhado):
    """
    Plota comparação entre demanda otimizada e não otimizada
    """
    # Calcular demanda não otimizada
    demanda_nao_otimizada = calcular_demanda_nao_otimizada(dados_aparelhos)
    pico_nao_otimizado = max(demanda_nao_otimizada)
    
    # Calcular demanda otimizada
    demanda_otimizada = [0] * 24
    
    if agendamento_detalhado:
        print("\n=== CONSTRUÇÃO DEMANDA OTIMIZADA ===")
        for tarefa_id, dados in agendamento_detalhado.items():
            inicio = dados['inicio']
            duracao = int(dados['duracao'])
            potencia = dados['potencia']
            
            print(f"{tarefa_id}: início={inicio}h, duração={duracao}h, potência={potencia}W")
            
            for h in range(duracao):
                hora_atual = (inicio + h) % 24
                demanda_otimizada[hora_atual] += potencia
                print(f"  Hora {hora_atual}: +{potencia}W = {demanda_otimizada[hora_atual]}W")
    
    pico_otimizado = max(demanda_otimizada)
    
    # Exibir dados no terminal
    print(f"\n=== RESULTADOS FINAIS ===")
    print(f"🔥 Pico NÃO otimizado: {pico_nao_otimizado/1000:.2f} kW")
    print(f"⚡ Pico otimizado: {pico_otimizado/1000:.2f} kW")
    print(f"💰 Redução do pico: {((pico_nao_otimizado - pico_otimizado)/pico_nao_otimizado)*100:.1f}%")
    
    print(f"\n📈 Demanda NÃO otimizada por hora (kW):")
    for h in range(24):
        print(f"  {h:2d}h: {demanda_nao_otimizada[h]/1000:6.2f} kW")
    
    print(f"\n📉 Demanda otimizada por hora (kW):")
    for h in range(24):
        print(f"  {h:2d}h: {demanda_otimizada[h]/1000:6.2f} kW")

    # Plotar gráfico
    horas = list(range(24))
    
    plt.figure(figsize=(15, 8))
    
    # Converter para kW
    demanda_nao_opt_kw = [d/1000 for d in demanda_nao_otimizada]
    demanda_opt_kw = [d/1000 for d in demanda_otimizada]
    
    plt.plot(horas, demanda_nao_opt_kw,
             label=f'Demanda Não Otimizada (Pico: {pico_nao_otimizado/1000:.2f} kW)',
             color='red', linestyle='--', linewidth=2, marker='o', markersize=4)
    
    plt.plot(horas, demanda_opt_kw,
             label=f'Demanda Otimizada (Pico: {pico_otimizado/1000:.2f} kW)',
             color='green', linewidth=3, marker='s', markersize=4)

    plt.title("Comparação: Demanda Elétrica com e sem Otimização", fontsize=16, fontweight='bold')
    plt.xlabel("Hora do Dia", fontsize=12)
    plt.ylabel("Demanda (kW)", fontsize=12)
    plt.xticks(range(0, 24, 2))
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=11)
    plt.tight_layout()
    
    # Salvar gráfico
    plt.savefig('comparacao_demanda.png', dpi=300, bbox_inches='tight')
    print(f"\n💾 Gráfico salvo como 'comparacao_demanda.png'")
    
    # Tentar mostrar (pode não funcionar em ambiente sem display)
    try:
        plt.show()
    except:
        print("⚠️  Display não disponível para mostrar gráfico interativo")


if __name__ == "__main__":
    print("🚀 Iniciando otimização de agendamento de cargas residenciais...\n")
    
    df_agendamento, pico, dados, agendamento_detalhado = otimizar_agendamento_residencial()
    
    if df_agendamento is not None:
        plotar_resultados(df_agendamento, dados, agendamento_detalhado)
        print(f"\n✅ Otimização concluída com sucesso!")
    else:
        print(f"\n❌ Falha na otimização!")