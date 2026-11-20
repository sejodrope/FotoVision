# Roteiro da defesa (TCC-II, 08/12/2026). Fonte única das falas: o gerador dos slides
# (build_slides.py) grava cada fala nas anotações do orador, e gera_roteiro_md.py monta o
# roteiro em Markdown. Números conferidos com o artigo v14 e a FICHA_TECNICA_FITOVISION.md.
# 19/11/2026: sem Q1–Q3 e sem objetivo geral/específicos (pedido do orientador: no TCC II há uma
# questão de pesquisa e um objetivo, iguais em todo o trabalho). Versão anterior: roteiro_data_v13_backup.py.
# Campos: secao (rótulo do modelo), titulo, tempo (segundos), pagina (no artigo v14),
# mensagem (a ideia única do slide), fala, transicao, se_perguntarem.

SLIDES = [
    dict(
        id='capa', secao='', titulo='Capa', tempo=30, pagina='',
        mensagem='Quem sou, o que é o trabalho e a pergunta em uma frase.',
        fala=('Boa tarde a todos. Meu nome é José Pedro, e este é o meu Trabalho de Conclusão de Curso em '
              'Engenharia de Software, orientado pelo professor Leanderson André. O título é “Detecção de anomalias '
              'fitossanitárias em hortaliças utilizando visão computacional e aprendizado de máquina”. Em uma frase: eu quis saber se um modelo que acerta 99% no computador continua acertando '
              'na horta.'),
        transicao='Começo pelo problema.',
        se_perguntarem='',
    ),
    dict(
        id='problema', secao='Introdução', titulo='99% no papel, 70% a 85% no campo', tempo=75, pagina='p. 1',
        mensagem='Existe uma distância entre o resultado de laboratório e o de campo, e ela tem custo para o produtor.',
        fala=('Diagnosticar doença de planta por foto é uma das aplicações mais estudadas de aprendizado profundo '
              'na agricultura. O artigo que abriu a área, em 2016, passou de 99% de acurácia. Só que aquelas fotos '
              'são de estúdio: folha isolada, fundo liso, luz controlada. Quando os mesmos modelos vão para a '
              'lavoura, com solo, sombra e várias plantas no quadro, a literatura registra entre 70% e 85%. Essa '
              'diferença tem nome: domain gap. E isso importa no Brasil, porque 77% dos estabelecimentos rurais são '
              'de agricultura familiar, onde o diagnóstico ainda é feito no olho. Em hortaliças folhosas, como '
              'alface, rúcula e couve, o ciclo é curto, e uma semana de atraso já vira perda. O risco não é só o '
              'modelo errar. É errar com cara de certeza, porque aí o produtor confia.'),
        transicao='A minha pergunta parte daí.',
        se_perguntarem=('De onde vêm os números? Mais de 99%: Mohanty, Hughes e Salathé (2016), no PlantVillage. '
                        '70% a 85%: revisão de Shafay et al. (2025). 77%: Censo Agropecuário 2017 (IBGE, 2019).'),
    ),
    dict(
        id='questao', secao='Introdução', titulo='Questão de pesquisa e objetivo', tempo=75, pagina='p. 2',
        mensagem='Uma questão de pesquisa e um objetivo, avaliados em três situações cada vez mais longe do treino.',
        fala=('A questão de pesquisa foi: em que medida um sistema de triagem fitossanitária para hortaliças '
              'folhosas, baseado em transfer learning, treinado com imagens públicas auditadas contra vazamento '
              'de dados e com calibração e abstenção, mantém desempenho confiável em fotografias de campo? O '
              'objetivo foi avaliar exatamente isso: em que medida esse sistema mantém desempenho confiável em '
              'fotografias de campo. Para responder, avaliei o sistema em três situações, cada uma mais distante '
              'das imagens de treino: um teste das mesmas fontes, uma fonte que o modelo nunca viu e fotos de '
              'horta. Em cada uma, olhei também se a abstenção pegava os erros. Já adianto: a resposta é parcial, '
              'e eu vou mostrar por quê.'),
        transicao='Primeiro, o que é o sistema.',
        se_perguntarem=('Objetivo geral e específicos eram do TCC-I; no TCC-II o trabalho tem uma questão e um '
                        'objetivo, os mesmos no resumo, na introdução, na metodologia e na conclusão. Se pedirem '
                        'o que foi feito: consolidar o dataset com deduplicação, comparar quatro arquiteturas, '
                        'avaliar calibração e abstenção e montar o conjunto de campo, que ainda não tem folhas doentes.'),
    ),
    dict(
        id='sistema', secao='Procedimentos metodológicos', titulo='O sistema: quatro respostas, não duas',
        tempo=75, pagina='p. 3 e 4',
        mensagem='É uma triagem com quatro veredictos; dois deles existem para o sistema poder dizer “não sei”.',
        fala=('O sistema recebe uma foto tirada com o celular e devolve uma de quatro respostas: saudável, '
              'anômalo, inconclusivo ou não é folha. As duas últimas existem porque uma rede binária sempre '
              'responde alguma coisa, até para a foto de uma parede. O “não é folha” vem de um índice de vegetação, '
              'o Excess Green: se menos de 10% dos pixels forem de vegetação, a foto é rejeitada antes de chegar à '
              'rede. O “inconclusivo” vem da abstenção: se a confiança calibrada ficar abaixo do limiar de 0,85, o '
              'sistema não dá diagnóstico. É uma triagem: ele diz se há algo errado com a folha, não qual é a '
              'doença. Esta tela é a interface que desenvolvi.'),
        transicao='Agora, os dados e o método.',
        se_perguntarem=('Por que binário? Porque as fontes públicas não têm as mesmas doenças nas mesmas culturas; '
                        'a triagem é a tarefa que os dados sustentam. Por que 0,85? É o menor limiar que garante '
                        '98% de acurácia seletiva na validação.'),
    ),
    dict(
        id='metodo', secao='Procedimentos metodológicos', titulo='Dados, modelos e avaliação', tempo=75,
        pagina='p. 4 a 6',
        mensagem='175 mil imagens deduplicadas, quatro arquiteturas, estatística pareada e três níveis de avaliação.',
        fala=('Os dados vêm de seis fontes públicas: 175 mil imagens, em duas classes. Para não repetir a mesma '
              'foto no treino e no teste, cada imagem vira um hash perceptual, as quase idênticas são agrupadas, e '
              'cada grupo vai inteiro para uma partição. Comparei quatro arquiteturas: três redes convolucionais, '
              'EfficientNet-B0, MobileNetV2 e ResNet-50, e um Vision Transformer, todas com transfer learning a '
              'partir do ImageNet. A métrica principal é a acurácia balanceada, com intervalo de confiança por '
              'bootstrap. Para comparar modelos, uso o teste de McNemar, porque todos são avaliados nas mesmas '
              'imagens. E a avaliação tem três níveis: o teste das mesmas fontes do treino, uma fonte nunca vista '
              'e fotos de campo, com o rótulo anotado antes de rodar o modelo.'),
        transicao='O primeiro resultado é sobre o meu próprio trabalho.',
        se_perguntarem=('Hash perceptual: resumo de 64 bits da imagem, invariante a rotação e espelhamento (grupo '
                        'D4); distância de Hamming até 4 = mesma foto. Treino: AdamW, até 30 épocas, 224×224, '
                        'semente 42. Partições da base: 120.861 / 25.944 / 23.967.'),
    ),
    dict(
        id='vazamento', secao='Resultados', titulo='O 99,01% estava errado', tempo=105, pagina='p. 2 e 4',
        mensagem='57,35% do teste do TCC-I já estava no treino. O número foi descartado.',
        fala=('No TCC-I eu cheguei a 99,01% de acurácia. Só que, em fotos novas, o sistema se comportava de forma '
              'errática. A pergunta que mudou o trabalho foi: por que esse número é bom demais? Fiz uma auditoria '
              'com hash perceptual e descobri que 57,35% do conjunto de teste eram fotos quase idênticas a fotos '
              'do treino: a mesma imagem recortada, girada ou comprimida. O modelo não estava generalizando; '
              'estava reconhecendo fotos que já tinha visto. Refiz a divisão agrupando as quase idênticas, e o '
              'vazamento caiu para 0,04%. O 99,01% foi descartado. Com o teste limpo, apareceram ainda dois '
              'defeitos que a métrica escondia: a abstenção nunca disparava, e quase 4 mil fotos de alface '
              'saudável estavam rotuladas como doentes, por causa do nome de uma pasta. Foram treze fotos de campo '
              'que denunciaram esse erro. Para mim, essa é a primeira lição do trabalho: desconfiar do número bom.'),
        transicao='Com o teste limpo, começo pela primeira situação: o teste das mesmas fontes.',
        se_perguntarem=('O dataset do TCC-I tinha 210.832 arquivos e só 106.403 fotografias distintas. O critério '
                        'atual ainda não pega variantes muito alteradas: 23% do teste são variantes de fotos de '
                        'alface com outra variante no treino (corrigido no ciclo 5: 1,7%).'),
    ),
    dict(
        id='q1', secao='Resultados', titulo='Arquiteturas · Empate no teste intra-fonte', tempo=90, pagina='p. 6',
        mensagem='EfficientNet-B0 e ViT-B/16 empataram; o ViT é 21 vezes maior e chegou ao mesmo lugar.',
        fala=('Primeiro, as arquiteturas. No teste, com quase 24 mil imagens, o EfficientNet-B0 '
              'e o ViT empatam em 98,48% de acurácia balanceada, com intervalo de confiança de 98,32 a 98,64. O '
              'teste de McNemar dá p igual a 0,26: não há diferença entre os dois. E os dois superam o MobileNetV2 '
              'e o ResNet-50, com p menor que dez elevado a menos onze. O detalhe importante: o ViT tem 21 vezes '
              'mais parâmetros que o EfficientNet e é quase três vezes mais lento. Chega ao mesmo resultado. Então, '
              'neste estudo, o EfficientNet-B0 foi a opção mais adequada: o mesmo desempenho com muito menos custo. '
              'E, nos meus dados, a arquitetura não foi o gargalo.'),
        transicao='Agora, a confiança: ela avisa quando o modelo erra?',
        se_perguntarem=('McNemar olha só as imagens em que os dois modelos discordam: 158 que só o EfficientNet '
                        'acerta, 180 que só o ViT acerta. Falsos negativos: EfficientNet 209, ViT 137. Há um '
                        'treino por modelo (semente 42): o empate pode mudar com outras sementes.'),
    ),
    dict(
        id='q2', secao='Resultados', titulo='Calibração · Calibrado na média, cego nos erros', tempo=90, pagina='p. 6',
        mensagem='A calibração funcionou, mas a abstenção interceptou só 31% e 15% dos erros.',
        fala=('A confiança do modelo avisa quando ele erra? Redes profundas costumam ser '
              'confiantes demais. Para corrigir, usei temperature scaling, e funcionou: o erro de calibração, o '
              'ECE, ficou em 0,003 na validação. Ou seja, quando o modelo diz 90%, ele acerta perto de 90%. Mas '
              'calibrar na média não basta. O limiar de abstenção intercepta só 31% dos erros do EfficientNet e '
              '15% dos erros do ViT. Mais de dois terços dos erros passam com confiança alta. E fora das fontes de '
              'treino piora: nos erros nas 13 fotos de campo, a confiança ficou entre 0,95 e 0,999, e o ECE subiu de '
              '0,003 para 0,146, um valor só indicativo com 13 fotos. Ou seja: a confiança ficou bem calibrada onde o modelo foi treinado, mas, '
              'sozinha, a abstenção não protegeu o produtor onde mais importa.'),
        transicao='E por que piora fora das fontes de treino? É o próximo slide.',
        se_perguntarem=('Temperature scaling divide os logits por um escalar T ajustado na validação (T = 0,561 no '
                        'EfficientNet). ECE: diferença média entre confiança e acerto, em 15 faixas. EfficientNet: '
                        '113 de 363 erros interceptados, abstenção em 1,2% das imagens. ViT: 51 de 341, 0,5%.'),
    ),
    dict(
        id='central', secao='Resultados', titulo='O teste por dentro e por fora', tempo=105, pagina='p. 6 e 7',
        mensagem='O 98,48% é uma média: 99,7% em estúdio, 86,3% em lavoura e 75,4% em uma fonte nunca vista.',
        fala=('Este é o resultado central do trabalho. O 98,48% é uma média, e a média esconde coisa. Quando '
              'separo o teste pelo tipo de foto, as culturas fotografadas em estúdio têm 99,7%. As fotografadas em '
              'lavoura, que são mandioca, arroz, cana e café, têm 86,3% e concentram 81% dos erros. Isso dentro do '
              'mesmo teste. E fui além: testei o modelo em uma fonte que ele nunca tinha visto, um lote de '
              'espinafre-de-malabar. A acurácia foi de 75,4%, 23 pontos abaixo, e eram fotos de folha isolada em '
              'fundo claro, parecidas com as do treino. Quando coloquei esse lote no treino, subiu para 88,4%, mas '
              'aí ele deixa de ser fonte nova. O que isso mostra: um teste sem duplicatas continua sendo um teste '
              'da mesma distribuição. O número do teste diz como o modelo vai nas fontes que conhece, não no mundo.'),
        transicao='Fui atrás da causa desse erro na lavoura.',
        se_perguntarem=('Estúdio: N = 10.061; lavoura: N = 4.530 (295 dos 363 erros). Espinafre: N = 484, modelo '
                        'do ciclo 3. É Basella alba (espinafre-de-malabar), não Spinacia oleracea. Não diga '
                        '“laboratório” para o 98,48%: diga “intra-fonte”.'),
    ),
    dict(
        id='ciclo5', secao='Resultados', titulo='Corrigir o atalho tem custo', tempo=90, pagina='p. 7',
        mensagem='O ciclo 5 reduz falsos alarmes na lavoura e perde um pouco de sensibilidade: é outro ponto de operação.',
        fala=('Uma auditoria mostrou de onde vinha parte desse erro: atalhos nos dados. A cultura estava associada '
              'ao rótulo, e variantes da mesma foto de alface apareciam dos dois lados da divisão. No quinto ciclo '
              'corrigi isso e treinei as quatro arquiteturas de novo. Na lavoura, as três redes convolucionais '
              'ganharam de 7 a 12 pontos de especificidade, ou seja, menos falsos alarmes. O custo veio na '
              'sensibilidade, que caiu de 2 a 5 pontos: um pouco mais de folhas doentes passando como sadias. O '
              'ViT quase não mudou. Então o ciclo 5 não é um modelo melhor em tudo; é outro ponto de operação. '
              'Qual dos dois usar depende do custo de cada erro para o produtor. E no lote de espinafre que nenhum '
              'modelo viu, os quatro ficam entre 86% e 90%: dez pontos abaixo das fontes de treino.'),
        transicao='Falta a terceira situação: o campo.',
        se_perguntarem=('Por imagem, na lavoura, o ciclo 4 acerta mais (p ≤ 0,04), porque três em cada quatro '
                        'imagens ali são anômalas. Teste do ciclo 5: 18.413 imagens; 97,4% no EfficientNet, não '
                        'comparável ao 98,48% (o teste é outro). ViT: p = 0,66.'),
    ),
    dict(
        id='campo', secao='Resultados', titulo='Campo · O que foi possível medir', tempo=105, pagina='p. 6 e 7',
        mensagem='Fundo real não gerou falso alarme; a sensibilidade em campo ainda não foi medida.',
        fala=('E no campo? Tenho dois conjuntos. O primeiro são 13 fotos de alface, do TCC-I. O '
              'EfficientNet do ciclo 4 acerta 11 de 13, ou 84,6%. Mas o intervalo de confiança vai de 58% a 96%. '
              'Com 13 fotos, nenhuma versão se distingue de outra; a amostra serviu para revelar defeitos, não '
              'para escolher modelo. O segundo conjunto é de 27 de outubro: 10 fotos de canteiros de alface, em '
              'horta, com fundo real, isto é, solo, mato e vários pés no quadro. As quatro versões classificaram '
              'as 10 como sadias, com confiança acima de 0,98, inclusive em recortes da imagem. Ou seja: o fundo '
              'real não provocou falso alarme. Só que a horta inteira estava sadia. Sem folhas doentes, não '
              'consegui medir a sensibilidade em campo nem o tamanho do domain gap. Por isso a resposta, '
              'nesta parte, é parcial.'),
        transicao='Esse padrão aparece também em outros trabalhos.',
        se_perguntarem=('Wilson para 10 de 10: 72,2% a 100%. Lote de casa (18/10): um pé colhido, fotografado em '
                        'toalha branca; tudo saiu anômalo, inclusive 3 sadias. Não é campo, e folha e cena não se '
                        'separam. Meta de coleta: 100 a 300 fotos (Wilson de 15,5 pontos com 100).'),
    ),
    dict(
        id='literatura', secao='Discussão', titulo='O mesmo padrão em outras culturas', tempo=45, pagina='p. 3 e 7',
        mensagem='A queda fora da fonte é regra na literatura; a lacuna é reunir tudo isso em hortaliças folhosas.',
        fala=('Esse padrão não é só meu. Em videira, a mesma EfficientNet-B0 faz 73,8% em um conjunto externo. Em '
              'amêndoa, o F1 cai de 0,98 para 0,56 quando as fontes são separadas. E um estudo deste ano, do '
              'PlantVillage para fotos de campo, mostra a queda de 99,7% para 32%, com o modelo ainda confiante. O '
              'que não encontrei foi um estudo que reúna, em hortaliças folhosas, auditoria de vazamento, '
              'calibração, abstenção e teste fora da fonte. É essa a lacuna que o trabalho ocupa.'),
        transicao='Antes de concluir, as limitações.',
        se_perguntarem=('Videira: Ergün et al. (2026). Amêndoa: Workneh et al. (2026). PlantVillage → PlantDoc: '
                        'Xiang, Shi e Zhu (2026), ResNet-50, confiança média de 79,8% com 32% de acerto. '
                        'Espinafre-de-malabar em partição interna: 97,3% (Kabya et al., 2026).'),
    ),
    dict(
        id='limitacoes', secao='Discussão', titulo='Limitações do estudo', tempo=75, pagina='p. 7',
        mensagem='Cinco ressalvas, ditas por mim antes que a banca pergunte.',
        fala=('Antes da conclusão, as limitações, ditas por mim. Primeira: a evidência de campo é pequena, 23 '
              'fotos, só de alface, e sem folhas doentes no lote com fundo real. Segunda: treino e teste vêm das '
              'mesmas fontes públicas, e parte das culturas entra só como aproximação visual das hortaliças. '
              'Terceira: a calibração foi ajustada fora do campo, e a literatura mostra que, nessa passagem, a '
              'confiança continua acima da acurácia. Quarta: a tarefa é binária; o sistema não diz qual é a '
              'doença, e sintomas parecidos podem ter causas diferentes. Quinta: cada modelo foi treinado uma vez '
              'só; com outra semente, o empate entre EfficientNet e ViT pode não se repetir. Nenhuma delas '
              'invalida os resultados, mas todas delimitam até onde eles valem.'),
        transicao='Com isso, a conclusão.',
        se_perguntarem=('Fontes: (ii) Xu et al. (2024); (iii) Xiang, Shi e Zhu (2026); (iv) Barbedo (2016); '
                        '(v) Coakley e Gundersen (2026), 100 sementes por experimento em ResNet e ViT.'),
    ),
    dict(
        id='conclusao', secao='Conclusão', titulo='O que o trabalho responde', tempo=90, pagina='p. 8',
        mensagem='O desempenho dependeu mais do protocolo de avaliação do que da arquitetura; a entrega é o protocolo.',
        fala=('Voltando ao começo: modelos que passam de 99% em laboratório perdem desempenho no campo, e em '
              'hortaliças folhosas isso quase não tinha sido medido. O objetivo foi avaliar em que medida o meu '
              'sistema mantém desempenho confiável em fotografias de campo. Os resultados indicam que o desempenho '
              'dependeu mais do protocolo de avaliação do que da arquitetura: 98,48% nas fontes de treino, 75,4% em '
              'uma fonte nunca vista e 84,6% em 13 fotos de campo, com um intervalo largo. EfficientNet-B0 e ViT '
              'empataram, e o EfficientNet foi o mais adequado. A calibração funcionou no domínio de treino, mas a '
              'abstenção deixou passar mais de dois terços dos erros. No campo, nenhum falso alarme em horta sadia, '
              'e a sensibilidade ainda sem medida. Então a resposta à questão é parcial. A principal contribuição '
              'é o protocolo: auditoria de vazamento, avaliação em três situações e testes pareados, tudo '
              'reprodutível por script. Este trabalho não entrega um modelo pronto para o campo. Entrega a medida '
              'de quanto falta e o caminho para chegar lá.'),
        transicao='Para fechar, os próximos passos.',
        se_perguntarem=('Então o sistema não funciona? Funciona nas fontes em que foi treinado; fora delas, o '
                        'trabalho mede a perda em vez de escondê-la. Um 99% retratado e um 75% medido valem mais '
                        'do que um 99% sem auditoria.'),
    ),
    dict(
        id='futuros', secao='Conclusão', titulo='Trabalhos futuros', tempo=30, pagina='p. 8',
        mensagem='Cinco próximos passos, todos saídos das limitações.',
        fala=('Como trabalhos futuros: completar a coleta com folhas doentes e outras culturas; recalibrar a '
              'confiança com fotos de campo; testar modelos de fundação congelados; repetir o treino com várias '
              'sementes; e definir, com os produtores, o custo de cada tipo de erro. Muito obrigado. Fico à '
              'disposição para as perguntas.'),
        transicao='',
        se_perguntarem=('Modelos de fundação: Nguyen et al. (2026) chegam a 51% com DINOv2 congelado, contra 26% a '
                        '30% do ajuste fino completo, do PlantVillage para o PlantDoc.'),
    ),
]

TOTAL = sum(s['tempo'] for s in SLIDES)
