# FitoVision — Roteiro de falas da defesa (até 20 minutos)

### TCC-II · Engenharia de Software · Univille · Banca em 08/12/2026 · Versão 2, de 19/11/2026 (correções do orientador)

> **O que é este documento.** O texto do que falar em cada slide de `docs/Entregas Finais/TCC2_Apresentacao_FitoVision_v2.pptx`, com o tempo de cada um, a frase de transição e a cola para perguntas. As mesmas falas estão nas anotações do orador de cada slide. Os números são os do artigo v14 e da `FICHA_TECNICA_FITOVISION.md`.
>
> **Tempo.** A soma dá **19:15**, com cerca de 1798 palavras. Os tempos de cada slide já contam as pausas e a troca de slide: é um ritmo calmo, de umas 95 palavras por minuto. Se você falar mais depressa, termina entre 15 e 17 minutos, o que é bom: o limite é de 20.
>
> **Não decore palavra por palavra.** Decore a *mensagem única* de cada slide e a frase de transição. O texto serve para treinar o ritmo e para você ouvir como a ideia soa dita em voz alta.


---

## 1. Mapa da apresentação

| # | Slide | Tempo | Relógio ao terminar | Mensagem única |
|---|---|---|---|---|
| 1 | Capa | 0:30 | **0:30** | Quem sou, o que é o trabalho e a pergunta em uma frase. |
| 2 | 99% no papel, 70% a 85% no campo | 1:15 | **1:45** | Existe uma distância entre o resultado de laboratório e o de campo, e ela tem custo para o produtor. |
| 3 | Questão de pesquisa e objetivo | 1:15 | **3:00** | Uma questão de pesquisa e um objetivo, avaliados em três situações cada vez mais longe do treino. |
| 4 | O sistema: quatro respostas, não duas | 1:15 | **4:15** | É uma triagem com quatro veredictos; dois deles existem para o sistema poder dizer “não sei”. |
| 5 | Dados, modelos e avaliação | 1:15 | **5:30** | 175 mil imagens deduplicadas, quatro arquiteturas, estatística pareada e três níveis de avaliação. |
| 6 | O 99,01% estava errado | 1:45 | **7:15** | 57,35% do teste do TCC-I já estava no treino. O número foi descartado. |
| 7 | Arquiteturas · Empate no teste intra-fonte | 1:30 | **8:45** | EfficientNet-B0 e ViT-B/16 empataram; o ViT é 21 vezes maior e chegou ao mesmo lugar. |
| 8 | Calibração · Calibrado na média, cego nos erros | 1:30 | **10:15** | A calibração funcionou, mas a abstenção interceptou só 31% e 15% dos erros. |
| 9 | O teste por dentro e por fora | 1:45 | **12:00** | O 98,48% é uma média: 99,7% em estúdio, 86,3% em lavoura e 75,4% em uma fonte nunca vista. |
| 10 | Corrigir o atalho tem custo | 1:30 | **13:30** | O ciclo 5 reduz falsos alarmes na lavoura e perde um pouco de sensibilidade: é outro ponto de operação. |
| 11 | Campo · O que foi possível medir | 1:45 | **15:15** | Fundo real não gerou falso alarme; a sensibilidade em campo ainda não foi medida. |
| 12 | O mesmo padrão em outras culturas | 0:45 | **16:00** | A queda fora da fonte é regra na literatura; a lacuna é reunir tudo isso em hortaliças folhosas. |
| 13 | Limitações do estudo | 1:15 | **17:15** | Cinco ressalvas, ditas por mim antes que a banca pergunte. |
| 14 | O que o trabalho responde | 1:30 | **18:45** | O desempenho dependeu mais do protocolo de avaliação do que da arquitetura; a entrega é o protocolo. |
| 15 | Trabalhos futuros | 0:30 | **19:15** | Cinco próximos passos, todos saídos das limitações. |
| 16 | Obrigado | — | 19:15 | Fica na tela durante as perguntas |

### Três pontos de controle

- **Fim do slide 5 (método): 5:30.** Se passar de 6:00, encurte o slide 6: fale só do 57,35% e do 0,04%.
- **Fim do slide 9 (resultado central): 12:00.** Se passar de 13:00, pule a segunda metade do slide 10 (o lote de espinafre).
- **Fim do slide 13 (limitações): 17:15.** Se passar de 18:00, diga só o título das cinco limitações e vá para a conclusão.

### Se o tempo for cortado para 15 minutos

Tire o slide 12 (literatura) e junte o 10 ao 9 em uma frase: “corrigir os atalhos nos dados reduziu os falsos alarmes na lavoura e custou um pouco de sensibilidade”. Encurte o slide 6 para um minuto. Dá cerca de 15:30.


---

## 2. Falas, slide a slide

### Slide 1 — Capa

**Tempo:** 0:30 (de 0:00 a 0:30)  
**Mensagem única:** Quem sou, o que é o trabalho e a pergunta em uma frase.

> Boa tarde a todos. Meu nome é José Pedro, e este é o meu Trabalho de Conclusão de Curso em Engenharia de Software, orientado pelo professor Leanderson André. O título é “Detecção de anomalias fitossanitárias em hortaliças utilizando visão computacional e aprendizado de máquina”. Em uma frase: eu quis saber se um modelo que acerta 99% no computador continua acertando na horta.

**Transição:** “Começo pelo problema.”

### Slide 2 — 99% no papel, 70% a 85% no campo

**Tempo:** 1:15 (de 0:30 a 1:45) · seção “Introdução” · no artigo: p. 1  
**Mensagem única:** Existe uma distância entre o resultado de laboratório e o de campo, e ela tem custo para o produtor.

> Diagnosticar doença de planta por foto é uma das aplicações mais estudadas de aprendizado profundo na agricultura. O artigo que abriu a área, em 2016, passou de 99% de acurácia. Só que aquelas fotos são de estúdio: folha isolada, fundo liso, luz controlada. Quando os mesmos modelos vão para a lavoura, com solo, sombra e várias plantas no quadro, a literatura registra entre 70% e 85%. Essa diferença tem nome: domain gap. E isso importa no Brasil, porque 77% dos estabelecimentos rurais são de agricultura familiar, onde o diagnóstico ainda é feito no olho. Em hortaliças folhosas, como alface, rúcula e couve, o ciclo é curto, e uma semana de atraso já vira perda. O risco não é só o modelo errar. É errar com cara de certeza, porque aí o produtor confia.

**Transição:** “A minha pergunta parte daí.”

**Cola para perguntas:** De onde vêm os números? Mais de 99%: Mohanty, Hughes e Salathé (2016), no PlantVillage. 70% a 85%: revisão de Shafay et al. (2025). 77%: Censo Agropecuário 2017 (IBGE, 2019).

### Slide 3 — Questão de pesquisa e objetivo

**Tempo:** 1:15 (de 1:45 a 3:00) · seção “Introdução” · no artigo: p. 2  
**Mensagem única:** Uma questão de pesquisa e um objetivo, avaliados em três situações cada vez mais longe do treino.

> A questão de pesquisa foi: em que medida um sistema de triagem fitossanitária para hortaliças folhosas, baseado em transfer learning, treinado com imagens públicas auditadas contra vazamento de dados e com calibração e abstenção, mantém desempenho confiável em fotografias de campo? O objetivo foi avaliar exatamente isso: em que medida esse sistema mantém desempenho confiável em fotografias de campo. Para responder, avaliei o sistema em três situações, cada uma mais distante das imagens de treino: um teste das mesmas fontes, uma fonte que o modelo nunca viu e fotos de horta. Em cada uma, olhei também se a abstenção pegava os erros. Já adianto: a resposta é parcial, e eu vou mostrar por quê.

**Transição:** “Primeiro, o que é o sistema.”

**Cola para perguntas:** Objetivo geral e específicos eram do TCC-I; no TCC-II o trabalho tem uma questão e um objetivo, os mesmos no resumo, na introdução, na metodologia e na conclusão. Se pedirem o que foi feito: consolidar o dataset com deduplicação, comparar quatro arquiteturas, avaliar calibração e abstenção e montar o conjunto de campo, que ainda não tem folhas doentes.

### Slide 4 — O sistema: quatro respostas, não duas

**Tempo:** 1:15 (de 3:00 a 4:15) · seção “Procedimentos metodológicos” · no artigo: p. 3 e 4  
**Mensagem única:** É uma triagem com quatro veredictos; dois deles existem para o sistema poder dizer “não sei”.

> O sistema recebe uma foto tirada com o celular e devolve uma de quatro respostas: saudável, anômalo, inconclusivo ou não é folha. As duas últimas existem porque uma rede binária sempre responde alguma coisa, até para a foto de uma parede. O “não é folha” vem de um índice de vegetação, o Excess Green: se menos de 10% dos pixels forem de vegetação, a foto é rejeitada antes de chegar à rede. O “inconclusivo” vem da abstenção: se a confiança calibrada ficar abaixo do limiar de 0,85, o sistema não dá diagnóstico. É uma triagem: ele diz se há algo errado com a folha, não qual é a doença. Esta tela é a interface que desenvolvi.

**Transição:** “Agora, os dados e o método.”

**Cola para perguntas:** Por que binário? Porque as fontes públicas não têm as mesmas doenças nas mesmas culturas; a triagem é a tarefa que os dados sustentam. Por que 0,85? É o menor limiar que garante 98% de acurácia seletiva na validação.

### Slide 5 — Dados, modelos e avaliação

**Tempo:** 1:15 (de 4:15 a 5:30) · seção “Procedimentos metodológicos” · no artigo: p. 4 a 6  
**Mensagem única:** 175 mil imagens deduplicadas, quatro arquiteturas, estatística pareada e três níveis de avaliação.

> Os dados vêm de seis fontes públicas: 175 mil imagens, em duas classes. Para não repetir a mesma foto no treino e no teste, cada imagem vira um hash perceptual, as quase idênticas são agrupadas, e cada grupo vai inteiro para uma partição. Comparei quatro arquiteturas: três redes convolucionais, EfficientNet-B0, MobileNetV2 e ResNet-50, e um Vision Transformer, todas com transfer learning a partir do ImageNet. A métrica principal é a acurácia balanceada, com intervalo de confiança por bootstrap. Para comparar modelos, uso o teste de McNemar, porque todos são avaliados nas mesmas imagens. E a avaliação tem três níveis: o teste das mesmas fontes do treino, uma fonte nunca vista e fotos de campo, com o rótulo anotado antes de rodar o modelo.

**Transição:** “O primeiro resultado é sobre o meu próprio trabalho.”

**Cola para perguntas:** Hash perceptual: resumo de 64 bits da imagem, invariante a rotação e espelhamento (grupo D4); distância de Hamming até 4 = mesma foto. Treino: AdamW, até 30 épocas, 224×224, semente 42. Partições da base: 120.861 / 25.944 / 23.967.

### Slide 6 — O 99,01% estava errado

**Tempo:** 1:45 (de 5:30 a 7:15) · seção “Resultados” · no artigo: p. 2 e 4  
**Mensagem única:** 57,35% do teste do TCC-I já estava no treino. O número foi descartado.

> No TCC-I eu cheguei a 99,01% de acurácia. Só que, em fotos novas, o sistema se comportava de forma errática. A pergunta que mudou o trabalho foi: por que esse número é bom demais? Fiz uma auditoria com hash perceptual e descobri que 57,35% do conjunto de teste eram fotos quase idênticas a fotos do treino: a mesma imagem recortada, girada ou comprimida. O modelo não estava generalizando; estava reconhecendo fotos que já tinha visto. Refiz a divisão agrupando as quase idênticas, e o vazamento caiu para 0,04%. O 99,01% foi descartado. Com o teste limpo, apareceram ainda dois defeitos que a métrica escondia: a abstenção nunca disparava, e quase 4 mil fotos de alface saudável estavam rotuladas como doentes, por causa do nome de uma pasta. Foram treze fotos de campo que denunciaram esse erro. Para mim, essa é a primeira lição do trabalho: desconfiar do número bom.

**Transição:** “Com o teste limpo, começo pela primeira situação: o teste das mesmas fontes.”

**Cola para perguntas:** O dataset do TCC-I tinha 210.832 arquivos e só 106.403 fotografias distintas. O critério atual ainda não pega variantes muito alteradas: 23% do teste são variantes de fotos de alface com outra variante no treino (corrigido no ciclo 5: 1,7%).

### Slide 7 — Arquiteturas · Empate no teste intra-fonte

**Tempo:** 1:30 (de 7:15 a 8:45) · seção “Resultados” · no artigo: p. 6  
**Mensagem única:** EfficientNet-B0 e ViT-B/16 empataram; o ViT é 21 vezes maior e chegou ao mesmo lugar.

> Primeiro, as arquiteturas. No teste, com quase 24 mil imagens, o EfficientNet-B0 e o ViT empatam em 98,48% de acurácia balanceada, com intervalo de confiança de 98,32 a 98,64. O teste de McNemar dá p igual a 0,26: não há diferença entre os dois. E os dois superam o MobileNetV2 e o ResNet-50, com p menor que dez elevado a menos onze. O detalhe importante: o ViT tem 21 vezes mais parâmetros que o EfficientNet e é quase três vezes mais lento. Chega ao mesmo resultado. Então, neste estudo, o EfficientNet-B0 foi a opção mais adequada: o mesmo desempenho com muito menos custo. E, nos meus dados, a arquitetura não foi o gargalo.

**Transição:** “Agora, a confiança: ela avisa quando o modelo erra?”

**Cola para perguntas:** McNemar olha só as imagens em que os dois modelos discordam: 158 que só o EfficientNet acerta, 180 que só o ViT acerta. Falsos negativos: EfficientNet 209, ViT 137. Há um treino por modelo (semente 42): o empate pode mudar com outras sementes.

### Slide 8 — Calibração · Calibrado na média, cego nos erros

**Tempo:** 1:30 (de 8:45 a 10:15) · seção “Resultados” · no artigo: p. 6  
**Mensagem única:** A calibração funcionou, mas a abstenção interceptou só 31% e 15% dos erros.

> A confiança do modelo avisa quando ele erra? Redes profundas costumam ser confiantes demais. Para corrigir, usei temperature scaling, e funcionou: o erro de calibração, o ECE, ficou em 0,003 na validação. Ou seja, quando o modelo diz 90%, ele acerta perto de 90%. Mas calibrar na média não basta. O limiar de abstenção intercepta só 31% dos erros do EfficientNet e 15% dos erros do ViT. Mais de dois terços dos erros passam com confiança alta. E fora das fontes de treino piora: nos erros nas 13 fotos de campo, a confiança ficou entre 0,95 e 0,999, e o ECE subiu de 0,003 para 0,146, um valor só indicativo com 13 fotos. Ou seja: a confiança ficou bem calibrada onde o modelo foi treinado, mas, sozinha, a abstenção não protegeu o produtor onde mais importa.

**Transição:** “E por que piora fora das fontes de treino? É o próximo slide.”

**Cola para perguntas:** Temperature scaling divide os logits por um escalar T ajustado na validação (T = 0,561 no EfficientNet). ECE: diferença média entre confiança e acerto, em 15 faixas. EfficientNet: 113 de 363 erros interceptados, abstenção em 1,2% das imagens. ViT: 51 de 341, 0,5%.

### Slide 9 — O teste por dentro e por fora

**Tempo:** 1:45 (de 10:15 a 12:00) · seção “Resultados” · no artigo: p. 6 e 7  
**Mensagem única:** O 98,48% é uma média: 99,7% em estúdio, 86,3% em lavoura e 75,4% em uma fonte nunca vista.

> Este é o resultado central do trabalho. O 98,48% é uma média, e a média esconde coisa. Quando separo o teste pelo tipo de foto, as culturas fotografadas em estúdio têm 99,7%. As fotografadas em lavoura, que são mandioca, arroz, cana e café, têm 86,3% e concentram 81% dos erros. Isso dentro do mesmo teste. E fui além: testei o modelo em uma fonte que ele nunca tinha visto, um lote de espinafre-de-malabar. A acurácia foi de 75,4%, 23 pontos abaixo, e eram fotos de folha isolada em fundo claro, parecidas com as do treino. Quando coloquei esse lote no treino, subiu para 88,4%, mas aí ele deixa de ser fonte nova. O que isso mostra: um teste sem duplicatas continua sendo um teste da mesma distribuição. O número do teste diz como o modelo vai nas fontes que conhece, não no mundo.

**Transição:** “Fui atrás da causa desse erro na lavoura.”

**Cola para perguntas:** Estúdio: N = 10.061; lavoura: N = 4.530 (295 dos 363 erros). Espinafre: N = 484, modelo do ciclo 3. É Basella alba (espinafre-de-malabar), não Spinacia oleracea. Não diga “laboratório” para o 98,48%: diga “intra-fonte”.

### Slide 10 — Corrigir o atalho tem custo

**Tempo:** 1:30 (de 12:00 a 13:30) · seção “Resultados” · no artigo: p. 7  
**Mensagem única:** O ciclo 5 reduz falsos alarmes na lavoura e perde um pouco de sensibilidade: é outro ponto de operação.

> Uma auditoria mostrou de onde vinha parte desse erro: atalhos nos dados. A cultura estava associada ao rótulo, e variantes da mesma foto de alface apareciam dos dois lados da divisão. No quinto ciclo corrigi isso e treinei as quatro arquiteturas de novo. Na lavoura, as três redes convolucionais ganharam de 7 a 12 pontos de especificidade, ou seja, menos falsos alarmes. O custo veio na sensibilidade, que caiu de 2 a 5 pontos: um pouco mais de folhas doentes passando como sadias. O ViT quase não mudou. Então o ciclo 5 não é um modelo melhor em tudo; é outro ponto de operação. Qual dos dois usar depende do custo de cada erro para o produtor. E no lote de espinafre que nenhum modelo viu, os quatro ficam entre 86% e 90%: dez pontos abaixo das fontes de treino.

**Transição:** “Falta a terceira situação: o campo.”

**Cola para perguntas:** Por imagem, na lavoura, o ciclo 4 acerta mais (p ≤ 0,04), porque três em cada quatro imagens ali são anômalas. Teste do ciclo 5: 18.413 imagens; 97,4% no EfficientNet, não comparável ao 98,48% (o teste é outro). ViT: p = 0,66.

### Slide 11 — Campo · O que foi possível medir

**Tempo:** 1:45 (de 13:30 a 15:15) · seção “Resultados” · no artigo: p. 6 e 7  
**Mensagem única:** Fundo real não gerou falso alarme; a sensibilidade em campo ainda não foi medida.

> E no campo? Tenho dois conjuntos. O primeiro são 13 fotos de alface, do TCC-I. O EfficientNet do ciclo 4 acerta 11 de 13, ou 84,6%. Mas o intervalo de confiança vai de 58% a 96%. Com 13 fotos, nenhuma versão se distingue de outra; a amostra serviu para revelar defeitos, não para escolher modelo. O segundo conjunto é de 27 de outubro: 10 fotos de canteiros de alface, em horta, com fundo real, isto é, solo, mato e vários pés no quadro. As quatro versões classificaram as 10 como sadias, com confiança acima de 0,98, inclusive em recortes da imagem. Ou seja: o fundo real não provocou falso alarme. Só que a horta inteira estava sadia. Sem folhas doentes, não consegui medir a sensibilidade em campo nem o tamanho do domain gap. Por isso a resposta, nesta parte, é parcial.

**Transição:** “Esse padrão aparece também em outros trabalhos.”

**Cola para perguntas:** Wilson para 10 de 10: 72,2% a 100%. Lote de casa (18/10): um pé colhido, fotografado em toalha branca; tudo saiu anômalo, inclusive 3 sadias. Não é campo, e folha e cena não se separam. Meta de coleta: 100 a 300 fotos (Wilson de 15,5 pontos com 100).

### Slide 12 — O mesmo padrão em outras culturas

**Tempo:** 0:45 (de 15:15 a 16:00) · seção “Discussão” · no artigo: p. 3 e 7  
**Mensagem única:** A queda fora da fonte é regra na literatura; a lacuna é reunir tudo isso em hortaliças folhosas.

> Esse padrão não é só meu. Em videira, a mesma EfficientNet-B0 faz 73,8% em um conjunto externo. Em amêndoa, o F1 cai de 0,98 para 0,56 quando as fontes são separadas. E um estudo deste ano, do PlantVillage para fotos de campo, mostra a queda de 99,7% para 32%, com o modelo ainda confiante. O que não encontrei foi um estudo que reúna, em hortaliças folhosas, auditoria de vazamento, calibração, abstenção e teste fora da fonte. É essa a lacuna que o trabalho ocupa.

**Transição:** “Antes de concluir, as limitações.”

**Cola para perguntas:** Videira: Ergün et al. (2026). Amêndoa: Workneh et al. (2026). PlantVillage → PlantDoc: Xiang, Shi e Zhu (2026), ResNet-50, confiança média de 79,8% com 32% de acerto. Espinafre-de-malabar em partição interna: 97,3% (Kabya et al., 2026).

### Slide 13 — Limitações do estudo

**Tempo:** 1:15 (de 16:00 a 17:15) · seção “Discussão” · no artigo: p. 7  
**Mensagem única:** Cinco ressalvas, ditas por mim antes que a banca pergunte.

> Antes da conclusão, as limitações, ditas por mim. Primeira: a evidência de campo é pequena, 23 fotos, só de alface, e sem folhas doentes no lote com fundo real. Segunda: treino e teste vêm das mesmas fontes públicas, e parte das culturas entra só como aproximação visual das hortaliças. Terceira: a calibração foi ajustada fora do campo, e a literatura mostra que, nessa passagem, a confiança continua acima da acurácia. Quarta: a tarefa é binária; o sistema não diz qual é a doença, e sintomas parecidos podem ter causas diferentes. Quinta: cada modelo foi treinado uma vez só; com outra semente, o empate entre EfficientNet e ViT pode não se repetir. Nenhuma delas invalida os resultados, mas todas delimitam até onde eles valem.

**Transição:** “Com isso, a conclusão.”

**Cola para perguntas:** Fontes: (ii) Xu et al. (2024); (iii) Xiang, Shi e Zhu (2026); (iv) Barbedo (2016); (v) Coakley e Gundersen (2026), 100 sementes por experimento em ResNet e ViT.

### Slide 14 — O que o trabalho responde

**Tempo:** 1:30 (de 17:15 a 18:45) · seção “Conclusão” · no artigo: p. 8  
**Mensagem única:** O desempenho dependeu mais do protocolo de avaliação do que da arquitetura; a entrega é o protocolo.

> Voltando ao começo: modelos que passam de 99% em laboratório perdem desempenho no campo, e em hortaliças folhosas isso quase não tinha sido medido. O objetivo foi avaliar em que medida o meu sistema mantém desempenho confiável em fotografias de campo. Os resultados indicam que o desempenho dependeu mais do protocolo de avaliação do que da arquitetura: 98,48% nas fontes de treino, 75,4% em uma fonte nunca vista e 84,6% em 13 fotos de campo, com um intervalo largo. EfficientNet-B0 e ViT empataram, e o EfficientNet foi o mais adequado. A calibração funcionou no domínio de treino, mas a abstenção deixou passar mais de dois terços dos erros. No campo, nenhum falso alarme em horta sadia, e a sensibilidade ainda sem medida. Então a resposta à questão é parcial. A principal contribuição é o protocolo: auditoria de vazamento, avaliação em três situações e testes pareados, tudo reprodutível por script. Este trabalho não entrega um modelo pronto para o campo. Entrega a medida de quanto falta e o caminho para chegar lá.

**Transição:** “Para fechar, os próximos passos.”

**Cola para perguntas:** Então o sistema não funciona? Funciona nas fontes em que foi treinado; fora delas, o trabalho mede a perda em vez de escondê-la. Um 99% retratado e um 75% medido valem mais do que um 99% sem auditoria.

### Slide 15 — Trabalhos futuros

**Tempo:** 0:30 (de 18:45 a 19:15) · seção “Conclusão” · no artigo: p. 8  
**Mensagem única:** Cinco próximos passos, todos saídos das limitações.

> Como trabalhos futuros: completar a coleta com folhas doentes e outras culturas; recalibrar a confiança com fotos de campo; testar modelos de fundação congelados; repetir o treino com várias sementes; e definir, com os produtores, o custo de cada tipo de erro. Muito obrigado. Fico à disposição para as perguntas.

**Cola para perguntas:** Modelos de fundação: Nguyen et al. (2026) chegam a 51% com DINOv2 congelado, contra 26% a 30% do ajuste fino completo, do PlantVillage para o PlantDoc.

### Slide 16 — Obrigado

Fique neste slide durante a arguição. Se a banca pedir um número, volte ao slide do resultado (7 para as arquiteturas, 8 para a calibração, 9 para o resultado central, 11 para o campo).


---

## 3. Regras de palco

- **Não leia o slide.** Cada slide tem um número grande; você conta a história desse número.
- **Diga os intervalos em voz alta.** “84,6%, com intervalo de 58 a 96” soa mais seguro do que “84,6%”.
- **Não diga “laboratório” para o 98,48%.** Diga “intra-fonte” ou “nas fontes de treino”.
- **Pare dois segundos depois do slide 6 e do slide 9.** São os dois momentos que a banca vai lembrar.
- **Limitações são ditas por você, sem pedir desculpas.** Um 99% retratado e um 75% medido formam um trabalho mais forte do que um 99% sem auditoria.
- **Se não souber responder:** “Não medi isso. O que tenho de mais próximo é … e o experimento que responderia seria …”. Inventar é o único erro grave.
- **Pronúncia:** *domain gap* (domêin gép), *temperature scaling* (témpratchur skêilin), McNemar (méc-nímar), ECE (ê-cê-é), ViT (vít), *bootstrap* (bútstrép).

## 4. Plano de treino

| Quando | O que fazer |
|---|---|
| Semana de 16/11 | Ler este roteiro em voz alta duas vezes, com cronômetro. Marcar onde trava. |
| Semana de 23/11 | Apresentar com os slides, sem ler. Gravar o áudio e ouvir. Decorar o cartão de números do guia. |
| Semana de 30/11 | Dois ensaios completos para alguém (um deles de fora da área). Treinar dez perguntas da Seção 11 do guia por dia. |
| 07/12 | Um ensaio só, de manhã. Conferir o arquivo no computador da sala, o adaptador e a cópia em PDF. |


## 5. Onde está o resto

- Guia de estudo completo (conceitos, literatura, 35 perguntas): `docs/GUIA_BANCA_TCC2.md`
- Artigo: `docs/Entregas Finais/TCC2_Artigo_FitoVision_v14.docx`
- Todos os números: `docs/FICHA_TECNICA_FITOVISION.md`
- Gerador dos slides e deste roteiro: `docs/apresentacao_banca/_gerador/`
