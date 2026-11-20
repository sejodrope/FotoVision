# Gera docs/apresentacao_banca/ROTEIRO_FALAS_20MIN.md a partir de roteiro_data.py.
# Uso: python gera_roteiro_md.py <saida.md>
import sys
from roteiro_data import SLIDES, TOTAL

def mmss(s): return f'{s // 60}:{s % 60:02d}'
def palavras(t): return len(t.split())

L = []
A = L.append
A('# FitoVision — Roteiro de falas da defesa (até 20 minutos)\n')
A('### TCC-II · Engenharia de Software · Univille · Banca em 08/12/2026 · Versão 2, de 19/11/2026 (correções do orientador)\n')
A('> **O que é este documento.** O texto do que falar em cada slide de '
  '`docs/Entregas Finais/TCC2_Apresentacao_FitoVision_v2.pptx`, com o tempo de cada um, a frase de transição e a '
  'cola para perguntas. As mesmas falas estão nas anotações do orador de cada slide. Os números são os do artigo '
  'v14 e da `FICHA_TECNICA_FITOVISION.md`.\n>\n'
  f'> **Tempo.** A soma dá **{mmss(TOTAL)}**, com cerca de {sum(palavras(s["fala"]) for s in SLIDES)} palavras. '
  'Os tempos de cada slide já contam as pausas e a troca de slide: é um ritmo calmo, de umas 95 palavras por '
  'minuto. Se você falar mais depressa, termina entre 15 e 17 minutos, o que é bom: o limite é de 20.\n>\n'
  '> **Não decore palavra por palavra.** Decore a *mensagem única* de cada slide e a frase de transição. O texto '
  'serve para treinar o ritmo e para você ouvir como a ideia soa dita em voz alta.\n')
A('\n---\n\n## 1. Mapa da apresentação\n')
A('| # | Slide | Tempo | Relógio ao terminar | Mensagem única |\n|---|---|---|---|---|')
ac = 0
for i, s in enumerate(SLIDES, 1):
    ac += s['tempo']
    A(f'| {i} | {s["titulo"]} | {mmss(s["tempo"])} | **{mmss(ac)}** | {s["mensagem"]} |')
A(f'| 16 | Obrigado | — | {mmss(ac)} | Fica na tela durante as perguntas |')
A('\n### Três pontos de controle\n')
def ate(id_):
    t = 0
    for s in SLIDES:
        t += s['tempo']
        if s['id'] == id_: return mmss(t)
A(f'- **Fim do slide 5 (método): {ate("metodo")}.** Se passar de 6:00, encurte o slide 6: fale só do 57,35% e do 0,04%.')
A(f'- **Fim do slide 9 (resultado central): {ate("central")}.** Se passar de 13:00, pule a segunda metade do slide 10 (o lote de espinafre).')
A(f'- **Fim do slide 13 (limitações): {ate("limitacoes")}.** Se passar de 18:00, diga só o título das cinco limitações e vá para a conclusão.')
A('\n### Se o tempo for cortado para 15 minutos\n')
A('Tire o slide 12 (literatura) e junte o 10 ao 9 em uma frase: “corrigir os atalhos nos dados reduziu os falsos '
  'alarmes na lavoura e custou um pouco de sensibilidade”. Encurte o slide 6 para um minuto. Dá cerca de 15:30.\n')
A('\n---\n\n## 2. Falas, slide a slide\n')
ac = 0
for i, s in enumerate(SLIDES, 1):
    ini = ac; ac += s['tempo']
    sec = f' · seção “{s["secao"]}”' if s['secao'] else ''
    A(f'### Slide {i} — {s["titulo"]}\n')
    A(f'**Tempo:** {mmss(s["tempo"])} (de {mmss(ini)} a {mmss(ac)}){sec}' + (f' · no artigo: {s["pagina"]}' if s['pagina'] else '') + '  ')
    A(f'**Mensagem única:** {s["mensagem"]}\n')
    A('> ' + s['fala'] + '\n')
    if s['transicao']: A(f'**Transição:** “{s["transicao"]}”\n')
    if s['se_perguntarem']: A(f'**Cola para perguntas:** {s["se_perguntarem"]}\n')
A('### Slide 16 — Obrigado\n')
A('Fique neste slide durante a arguição. Se a banca pedir um número, volte ao slide do resultado '
  '(7 para as arquiteturas, 8 para a calibração, 9 para o resultado central, 11 para o campo).\n')
A('\n---\n\n## 3. Regras de palco\n')
for r in [
    '**Não leia o slide.** Cada slide tem um número grande; você conta a história desse número.',
    '**Diga os intervalos em voz alta.** “84,6%, com intervalo de 58 a 96” soa mais seguro do que “84,6%”.',
    '**Não diga “laboratório” para o 98,48%.** Diga “intra-fonte” ou “nas fontes de treino”.',
    '**Pare dois segundos depois do slide 6 e do slide 9.** São os dois momentos que a banca vai lembrar.',
    '**Limitações são ditas por você, sem pedir desculpas.** Um 99% retratado e um 75% medido formam um trabalho mais forte do que um 99% sem auditoria.',
    '**Se não souber responder:** “Não medi isso. O que tenho de mais próximo é … e o experimento que responderia seria …”. Inventar é o único erro grave.',
    '**Pronúncia:** *domain gap* (domêin gép), *temperature scaling* (témpratchur skêilin), McNemar (méc-nímar), ECE (ê-cê-é), ViT (vít), *bootstrap* (bútstrép).',
]: A('- ' + r)
A('\n## 4. Plano de treino\n')
A('| Quando | O que fazer |\n|---|---|\n'
  '| Semana de 16/11 | Ler este roteiro em voz alta duas vezes, com cronômetro. Marcar onde trava. |\n'
  '| Semana de 23/11 | Apresentar com os slides, sem ler. Gravar o áudio e ouvir. Decorar o cartão de números do guia. |\n'
  '| Semana de 30/11 | Dois ensaios completos para alguém (um deles de fora da área). Treinar dez perguntas da Seção 11 do guia por dia. |\n'
  '| 07/12 | Um ensaio só, de manhã. Conferir o arquivo no computador da sala, o adaptador e a cópia em PDF. |\n')
A('\n## 5. Onde está o resto\n')
A('- Guia de estudo completo (conceitos, literatura, 35 perguntas): `docs/GUIA_BANCA_TCC2.md`\n'
  '- Artigo: `docs/Entregas Finais/TCC2_Artigo_FitoVision_v14.docx`\n'
  '- Todos os números: `docs/FICHA_TECNICA_FITOVISION.md`\n'
  '- Gerador dos slides e deste roteiro: `docs/apresentacao_banca/_gerador/`')
open(sys.argv[1], 'w', encoding='utf-8').write('\n'.join(L) + '\n')
print('ok', mmss(TOTAL), sum(palavras(s['fala']) for s in SLIDES))
