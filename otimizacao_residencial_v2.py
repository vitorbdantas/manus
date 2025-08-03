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
    print("Residência | Aparelho        | Potência (W) | Duração (h) | Janela Início | Janela Fim")
    print("-" * 95)
    for res_id, aparelho, potencia, dur_h, ini_h, fim_h in dados_aparelhos:
        print(f"    {res_id:2d}     | {aparelho:15s} | {potencia:8d}     | {dur_h:7.1f}     | {ini_h:9d}     | {fim_h:6d}")
    print()

    # Criar modelo
    modelo = cp_model.CpModel()
    
    # Variáveis: início de cada tarefa (0-23 horas)
    tarefas = []
    variaveis_inicio = {}
    
    for i, (res_id, aparelho, potencia, dur_h, ini_h, fim_h) in enumerate(dados_aparelhos):
        tarefa_id = f'T{i}_Res{res_id}_{aparelho.replace(" ", "_")}'
        
        # Ajustar janela se cruza meia-noite
        if fim_h <= ini_h:
            fim_h += 24
            
        # Calcular horários válidos de início
        max_inicio = fim_h - dur_h
        
        # Variável de início da tarefa
        inicio_var = modelo.NewIntVar(ini_h, max_inicio, f'{tarefa_id}_inicio')
        variaveis_inicio[tarefa_id] = inicio_var
        
        tarefas.append({
            'id': tarefa_id,
            'res_id': res_id,
            'aparelho': aparelho,
            'potencia': potencia,
            'duracao': int(dur_h),
            'janela_inicio': ini_h,
            'janela_fim': fim_h,
            'inicio_var': inicio_var
        })

    print(f"📋 Total de tarefas criadas: {len(tarefas)}")

    # Criar variáveis de demanda para cada hora
    demanda_hora = {}
    for h in range(24):
        demanda_hora[h] = modelo.NewIntVar(0, 200000, f'demanda_h{h}')

    # Para cada hora, calcular demanda total
    for h in range(24):
        termos_demanda = []
        
        for tarefa in tarefas:
            inicio_var = tarefa['inicio_var']
            duracao = tarefa['duracao']
            potencia = tarefa['potencia']
            
            # Para cada hora possível de operação desta tarefa
            for dur_offset in range(duracao):
                # Variável booleana: tarefa está ativa nesta hora específica?
                ativa_var = modelo.NewBoolVar(f"{tarefa['id']}_ativa_h{h}_offset{dur_offset}")
                
                # A tarefa está ativa se: hora atual = início + offset
                modelo.Add(inicio_var + dur_offset == h).OnlyEnforceIf(ativa_var)
                modelo.Add(inicio_var + dur_offset != h).OnlyEnforceIf(ativa_var.Not())
                
                # Contribuição para demanda
                termos_demanda.append(ativa_var * potencia)
        
        # Definir demanda total nesta hora
        modelo.Add(demanda_hora[h] == sum(termos_demanda))

    # Variável objetivo: pico de demanda
    pico_demanda = modelo.NewIntVar(0, 200000, 'pico')
    modelo.AddMaxEquality(pico_demanda, [demanda_hora[h] for h in range(24)])
    
    # Minimizar pico
    modelo.Minimize(pico_demanda)

    # Resolver
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 60.0
    solver.parameters.log_search_progress = True
    
    print("🔍 Resolvendo modelo de otimização...")
    status = solver.Solve(modelo)

    if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        print("✅ Solução encontrada!")
        pico_otimizado = solver.ObjectiveValue()
        print(f"🎯 Pico de Demanda Otimizado: {pico_otimizado / 1000:.2f} kW")
        
        # Extrair resultados
        agendamento = []
        agendamento_detalhado = {}
        
        for tarefa in tarefas:
            inicio_otimizado = solver.Value(tarefa['inicio_var'])
            fim_otimizado = inicio_otimizado + tarefa['duracao']
            
            agendamento.append({
                'Tarefa': tarefa['id'],
                'Residência': tarefa['res_id'],
                'Aparelho': tarefa['aparelho'],
                'Potência (W)': tarefa['potencia'],
                'Duração (h)': tarefa['duracao'],
                'Janela Início': tarefa['janela_inicio'],
                'Janela Fim': tarefa['janela_fim'],
                'Início Otimizado': inicio_otimizado % 24,
                'Fim Otimizado': fim_otimizado % 24
            })
            
            agendamento_detalhado[tarefa['id']] = {
                'inicio': inicio_otimizado % 24,
                'duracao': tarefa['duracao'],
                'potencia': tarefa['potencia']
            }

        df = pd.DataFrame(agendamento)
        print("\n=== AGENDAMENTO OTIMIZADO ===")
        print(df.to_string(index=False))
        
        return df, pico_otimizado, dados_aparelhos, agendamento_detalhado
    else:
        print("❌ Não foi possível encontrar uma solução.")
        print(f"Status: {solver.StatusName(status)}")
        return None, None, None, None


def calcular_demanda_nao_otimizada(dados_aparelhos):
    """
    Calcula a demanda sem otimização (horários originais)
    """
    demanda = [0] * 24

    print("\n=== CÁLCULO DEMANDA NÃO OTIMIZADA ===")
    for res_id, aparelho, potencia, dur_h, ini_h, fim_h in dados_aparelhos:
        print(f"Res{res_id} {aparelho}: {ini_h}h por {dur_h}h, {potencia}W")
        
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
            duracao = dados['duracao']
            potencia = dados['potencia']
            
            print(f"{tarefa_id}: início={inicio}h, duração={duracao}h, potência={potencia}W")
            
            for h in range(duracao):
                hora_atual = (inicio + h) % 24
                demanda_otimizada[hora_atual] += potencia
    
    pico_otimizado = max(demanda_otimizada) if demanda_otimizada else 0
    
    # Exibir resultados
    print(f"\n=== RESULTADOS FINAIS ===")
    print(f"🔥 Pico NÃO otimizado: {pico_nao_otimizado/1000:.2f} kW")
    print(f"⚡ Pico otimizado: {pico_otimizado/1000:.2f} kW")
    if pico_nao_otimizado > 0:
        reducao = ((pico_nao_otimizado - pico_otimizado)/pico_nao_otimizado)*100
        print(f"💰 Redução do pico: {reducao:.1f}%")
    
    # Tabela comparativa
    print(f"\n📊 COMPARAÇÃO HORA A HORA:")
    print("Hora | Não Otim. (kW) | Otimizada (kW) | Diferença (kW)")
    print("-" * 55)
    for h in range(24):
        nao_opt = demanda_nao_otimizada[h]/1000
        opt = demanda_otimizada[h]/1000
        diff = nao_opt - opt
        print(f" {h:2d}h |     {nao_opt:8.2f}   |    {opt:8.2f}   |    {diff:8.2f}")

    # Plotar gráfico
    horas = list(range(24))
    
    plt.figure(figsize=(16, 10))
    
    # Subplot 1: Comparação principal
    plt.subplot(2, 1, 1)
    demanda_nao_opt_kw = [d/1000 for d in demanda_nao_otimizada]
    demanda_opt_kw = [d/1000 for d in demanda_otimizada]
    
    plt.plot(horas, demanda_nao_opt_kw,
             label=f'Demanda Não Otimizada (Pico: {pico_nao_otimizado/1000:.2f} kW)',
             color='red', linestyle='--', linewidth=3, marker='o', markersize=5)
    
    plt.plot(horas, demanda_opt_kw,
             label=f'Demanda Otimizada (Pico: {pico_otimizado/1000:.2f} kW)',
             color='green', linewidth=3, marker='s', markersize=5)

    plt.title("Comparação: Demanda Elétrica com e sem Otimização", fontsize=16, fontweight='bold')
    plt.xlabel("Hora do Dia", fontsize=12)
    plt.ylabel("Demanda (kW)", fontsize=12)
    plt.xticks(range(0, 24, 2))
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=11)
    
    # Subplot 2: Diferença
    plt.subplot(2, 1, 2)
    diferenca = [demanda_nao_opt_kw[h] - demanda_opt_kw[h] for h in range(24)]
    colors = ['green' if d > 0 else 'red' for d in diferenca]
    
    plt.bar(horas, diferenca, color=colors, alpha=0.7)
    plt.title("Redução/Aumento de Demanda por Hora", fontsize=14, fontweight='bold')
    plt.xlabel("Hora do Dia", fontsize=12)
    plt.ylabel("Diferença (kW)", fontsize=12)
    plt.xticks(range(0, 24, 2))
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.axhline(y=0, color='black', linestyle='-', linewidth=1)
    
    plt.tight_layout()
    
    # Salvar gráfico
    plt.savefig('comparacao_demanda_v2.png', dpi=300, bbox_inches='tight')
    print(f"\n💾 Gráfico salvo como 'comparacao_demanda_v2.png'")
    
    # Tentar mostrar
    try:
        plt.show()
    except:
        print("⚠️  Display não disponível para mostrar gráfico interativo")


if __name__ == "__main__":
    print("🚀 Iniciando otimização avançada de agendamento residencial...\n")
    
    df_agendamento, pico, dados, agendamento_detalhado = otimizar_agendamento_residencial()
    
    if df_agendamento is not None:
        plotar_resultados(df_agendamento, dados, agendamento_detalhado)
        print(f"\n✅ Otimização concluída com sucesso!")
    else:
        print(f"\n❌ Falha na otimização!")