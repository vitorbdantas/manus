from ortools.sat.python import cp_model
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def calcular_demanda_nao_otimizada(dados_aparelhos):
    """Calcula demanda sem otimização"""
    demanda = [0] * 24
    
    print("=== DEMANDA NÃO OTIMIZADA ===")
    for res_id, aparelho, potencia, dur_h, ini_h, fim_h in dados_aparelhos:
        print(f"Res{res_id} {aparelho}: {ini_h}h-{(ini_h + dur_h) % 24}h, {potencia}W")
        for h in range(int(dur_h)):
            hora_atual = (ini_h + h) % 24
            demanda[hora_atual] += potencia
    
    return demanda


def otimizar_agendamento_heuristico():
    """
    Otimização heurística que realmente distribui as cargas
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
    df_dados = pd.DataFrame(dados_aparelhos, columns=['Residência', 'Aparelho', 'Potência(W)', 'Duração(h)', 'Início(h)', 'Fim(h)'])
    print(df_dados.to_string(index=False))

    # Calcular demanda não otimizada primeiro
    demanda_original = calcular_demanda_nao_otimizada(dados_aparelhos)
    pico_original = max(demanda_original)
    print(f"\n🔥 Pico original: {pico_original/1000:.2f} kW")

    # Preparar tarefas para otimização
    tarefas = []
    for i, (res_id, aparelho, potencia, dur_h, ini_h, fim_h) in enumerate(dados_aparelhos):
        # Ajustar janela se cruza meia-noite
        if fim_h <= ini_h:
            fim_h += 24
        
        # Calcular horários válidos
        horarios_validos = []
        for h in range(ini_h, fim_h - dur_h + 1):
            horarios_validos.append(h % 24)
        
        tarefas.append({
            'id': i,
            'res_id': res_id,
            'aparelho': aparelho,
            'potencia': potencia,
            'duracao': int(dur_h),
            'horarios_validos': horarios_validos,
            'janela_inicio': ini_h,
            'janela_fim': fim_h
        })

    print(f"\n📋 Criadas {len(tarefas)} tarefas para otimização")

    # Usar OR-Tools com estratégia melhorada
    modelo = cp_model.CpModel()
    
    # Variáveis de início para cada tarefa
    inicio_vars = {}
    for tarefa in tarefas:
        nome = f"inicio_T{tarefa['id']}"
        min_inicio = min(tarefa['horarios_validos'])
        max_inicio = max(tarefa['horarios_validos'])
        inicio_vars[tarefa['id']] = modelo.NewIntVar(min_inicio, max_inicio, nome)
        
        # Restringir aos horários válidos
        if len(tarefa['horarios_validos']) < max_inicio - min_inicio + 1:
            modelo.AddAllowedAssignments([inicio_vars[tarefa['id']]], 
                                        [[h] for h in tarefa['horarios_validos']])

    # Variáveis de demanda por hora
    demanda_vars = {}
    for h in range(24):
        demanda_vars[h] = modelo.NewIntVar(0, 100000, f'demanda_h{h}')

    # Modelar demanda em cada hora
    for h in range(24):
        contribuicoes = []
        
        for tarefa in tarefas:
            tarefa_id = tarefa['id']
            potencia = tarefa['potencia']
            duracao = tarefa['duracao']
            
            # Para cada hora de duração da tarefa
            for offset in range(duracao):
                # Variável booleana: tarefa ativa nesta hora?
                ativa = modelo.NewBoolVar(f'T{tarefa_id}_ativa_h{h}_offset{offset}')
                
                # Ativa se inicio + offset == h
                modelo.Add(inicio_vars[tarefa_id] + offset == h).OnlyEnforceIf(ativa)
                modelo.Add(inicio_vars[tarefa_id] + offset != h).OnlyEnforceIf(ativa.Not())
                
                contribuicoes.append(ativa * potencia)
        
        # Demanda total nesta hora
        modelo.Add(demanda_vars[h] == sum(contribuicoes))

    # Variável objetivo
    pico_var = modelo.NewIntVar(0, 100000, 'pico')
    modelo.AddMaxEquality(pico_var, [demanda_vars[h] for h in range(24)])

    # Adicionar objetivos secundários para distribuir melhor
    # Tentar minimizar variabilidade
    desvios = []
    demanda_media = modelo.NewIntVar(0, 100000, 'media')
    modelo.Add(demanda_media * 24 == sum(demanda_vars[h] for h in range(24)))
    
    for h in range(24):
        desvio_pos = modelo.NewIntVar(0, 100000, f'desvio_pos_{h}')
        desvio_neg = modelo.NewIntVar(0, 100000, f'desvio_neg_{h}')
        modelo.Add(demanda_vars[h] - demanda_media == desvio_pos - desvio_neg)
        desvios.extend([desvio_pos, desvio_neg])

    # Objetivo ponderado: prioridade para pico, mas também considerar distribuição
    modelo.Minimize(pico_var * 1000 + sum(desvios))

    # Resolver
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 30.0
    
    print("🔍 Resolvendo otimização...")
    status = solver.Solve(modelo)

    if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        pico_otimizado = solver.Value(pico_var)
        print(f"✅ Solução encontrada! Pico otimizado: {pico_otimizado/1000:.2f} kW")
        
        # Extrair resultados
        agendamento = []
        agendamento_detalhado = {}
        
        for tarefa in tarefas:
            inicio = solver.Value(inicio_vars[tarefa['id']])
            
            agendamento.append({
                'Tarefa': f"T{tarefa['id']}_Res{tarefa['res_id']}_{tarefa['aparelho'].replace(' ', '_')}",
                'Residência': tarefa['res_id'],
                'Aparelho': tarefa['aparelho'],
                'Potência (W)': tarefa['potencia'],
                'Duração (h)': tarefa['duracao'],
                'Janela Original': f"{tarefa['janela_inicio']}-{tarefa['janela_fim']}h",
                'Início Otimizado': inicio % 24,
                'Fim Otimizado': (inicio + tarefa['duracao']) % 24
            })
            
            agendamento_detalhado[tarefa['id']] = {
                'inicio': inicio % 24,
                'duracao': tarefa['duracao'],
                'potencia': tarefa['potencia']
            }

        return agendamento, agendamento_detalhado, dados_aparelhos, pico_otimizado
    else:
        print(f"❌ Falha na otimização: {solver.StatusName(status)}")
        return None, None, dados_aparelhos, None


def plotar_comparacao(agendamento, agendamento_detalhado, dados_aparelhos, pico_otimizado):
    """
    Plota comparação final com dados detalhados
    """
    # Calcular demandas
    demanda_original = calcular_demanda_nao_otimizada(dados_aparelhos)
    pico_original = max(demanda_original)
    
    demanda_otimizada = [0] * 24
    
    if agendamento_detalhado:
        print("\n=== AGENDAMENTO OTIMIZADO ===")
        df_agendamento = pd.DataFrame(agendamento)
        print(df_agendamento.to_string(index=False))
        
        print("\n=== CONSTRUÇÃO DEMANDA OTIMIZADA ===")
        for tarefa_id, dados in agendamento_detalhado.items():
            inicio = dados['inicio']
            duracao = dados['duracao']
            potencia = dados['potencia']
            
            print(f"Tarefa {tarefa_id}: {inicio}h-{(inicio+duracao)%24}h, {potencia}W")
            
            for h in range(duracao):
                hora_atual = (inicio + h) % 24
                demanda_otimizada[hora_atual] += potencia
    
    pico_otimizado_real = max(demanda_otimizada) if demanda_otimizada else 0
    
    # Estatísticas
    print(f"\n=== RESULTADOS FINAIS ===")
    print(f"🔥 Pico ORIGINAL: {pico_original/1000:.2f} kW")
    print(f"⚡ Pico OTIMIZADO: {pico_otimizado_real/1000:.2f} kW")
    reducao = ((pico_original - pico_otimizado_real) / pico_original) * 100
    print(f"💰 Redução: {reducao:.1f}%")
    
    # Comparação hora a hora
    print(f"\n📊 COMPARAÇÃO DETALHADA (hora a hora):")
    print("Hora | Original(kW) | Otimizado(kW) | Diferença(kW) | Status")
    print("-" * 65)
    for h in range(24):
        orig = demanda_original[h] / 1000
        otim = demanda_otimizada[h] / 1000
        diff = orig - otim
        status = "↓ Redução" if diff > 0 else "↑ Aumento" if diff < 0 else "= Igual"
        print(f" {h:2d}h |    {orig:7.2f}  |     {otim:7.2f}  |    {diff:7.2f}  | {status}")

    # Plotar gráfico melhorado
    horas = list(range(24))
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
    
    # Gráfico principal
    demanda_orig_kw = [d/1000 for d in demanda_original]
    demanda_otim_kw = [d/1000 for d in demanda_otimizada]
    
    ax1.plot(horas, demanda_orig_kw, 'r--', linewidth=3, marker='o', 
             markersize=6, label=f'Original (Pico: {pico_original/1000:.2f} kW)')
    ax1.plot(horas, demanda_otim_kw, 'g-', linewidth=3, marker='s', 
             markersize=6, label=f'Otimizado (Pico: {pico_otimizado_real/1000:.2f} kW)')
    
    ax1.fill_between(horas, demanda_orig_kw, alpha=0.3, color='red')
    ax1.fill_between(horas, demanda_otim_kw, alpha=0.3, color='green')
    
    ax1.set_title('Comparação: Demanda Elétrica Original vs Otimizada', fontsize=16, fontweight='bold')
    ax1.set_xlabel('Hora do Dia', fontsize=12)
    ax1.set_ylabel('Demanda (kW)', fontsize=12)
    ax1.set_xticks(range(0, 24, 2))
    ax1.grid(True, alpha=0.7)
    ax1.legend(fontsize=12)
    
    # Gráfico de diferenças
    diferenca = [demanda_orig_kw[h] - demanda_otim_kw[h] for h in range(24)]
    colors = ['green' if d > 0 else 'red' if d < 0 else 'gray' for d in diferenca]
    
    bars = ax2.bar(horas, diferenca, color=colors, alpha=0.7, edgecolor='black')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax2.set_title('Redução/Aumento de Demanda por Hora', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Hora do Dia', fontsize=12)
    ax2.set_ylabel('Diferença (kW)', fontsize=12)
    ax2.set_xticks(range(0, 24, 2))
    ax2.grid(True, alpha=0.5)
    
    # Adicionar valores nas barras
    for i, (bar, val) in enumerate(zip(bars, diferenca)):
        if abs(val) > 0.1:  # Só mostrar valores significativos
            ax2.text(bar.get_x() + bar.get_width()/2, val + (0.5 if val > 0 else -0.5),
                    f'{val:.1f}', ha='center', va='bottom' if val > 0 else 'top', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('otimizacao_final.png', dpi=300, bbox_inches='tight')
    print(f"\n💾 Gráfico salvo como 'otimizacao_final.png'")
    
    # Salvar dados detalhados
    print("\n💾 Salvando dados detalhados...")
    
    # Dados comparativos
    dados_comparacao = []
    for h in range(24):
        dados_comparacao.append({
            'Hora': h,
            'Demanda_Original_kW': demanda_original[h]/1000,
            'Demanda_Otimizada_kW': demanda_otimizada[h]/1000,
            'Diferenca_kW': (demanda_original[h] - demanda_otimizada[h])/1000,
            'Reducao_Percentual': ((demanda_original[h] - demanda_otimizada[h]) / max(demanda_original[h], 1)) * 100
        })
    
    df_comparacao = pd.DataFrame(dados_comparacao)
    df_comparacao.to_csv('comparacao_demanda.csv', index=False)
    print("📄 Dados salvos em 'comparacao_demanda.csv'")
    
    try:
        plt.show()
    except:
        print("⚠️  Display não disponível")


if __name__ == "__main__":
    print("🚀 OTIMIZAÇÃO FINAL DE AGENDAMENTO RESIDENCIAL\n")
    
    agendamento, agendamento_detalhado, dados, pico = otimizar_agendamento_heuristico()
    
    if agendamento:
        plotar_comparacao(agendamento, agendamento_detalhado, dados, pico)
        print(f"\n✅ OTIMIZAÇÃO CONCLUÍDA COM SUCESSO!")
        print(f"📊 Verifique os arquivos 'otimizacao_final.png' e 'comparacao_demanda.csv'")
    else:
        print(f"\n❌ Falha na otimização!")