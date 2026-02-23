# Planejamento de atualização do repo minutiaenet

Eu tenho um repositório chamado minutiaenet que é código aberto mas usa tecnologias antigas:

```
Python 2.7, Tensorflow 1.7.0, Keras 2.1.6.
```

Eu quero atualizar isso para código python 3.1x e pytorch.

Eu fiz isso já com o repositório fingernet.

Eu adicionei:
- Uma api para rodar no python com facilidade, importanto o módulo
- Uma CLI para rodar inferência em múltiplas gpus e realizar extrações em lote de larga escala
- Um dockerfile para rodar a versão antiga e verificar a equivalência com a nova versão
- Reescrita do modelo de forma modular
- Funções de plot para visualizar os resultados
- Alguns exemplos de uso

# A sua missão

Você tem como missão atualizar o repositório minutiaenet da mesma forma. As suas primeiras tarefas serão:

1) Converter os modelos, refatorando o modelo de forma modular .h5 para para .pth
2) Criar uma API e CLI para rodar o modelo com facilidade
    a) Criar a interface com DDP para rodar em múltiplas gpus
    b) Refatorar para que o código seja importado como um módulo python

Os requisitos:
1) Use o ambiente conda chamado `grids`, lá eu tenho os pacotes instalados.
2) Caso você precise de algum pacote adicional/versão diferente, me avise para que eu possa instalar.
3) O código deve ser escrito de forma clara e modular, seguindo boas práticas de programação.
4) Qualquer decisão que diverge do código original deve ser avisada a mim para que eu possa aprovar ou sugerir mudanças.
5) O código deve ser testado para garantir que a nova implementação é equivalente à antiga, as imagens de entrada podem ser encontradas em `/home/joaocontreras/work/fingernet/datasets`

# Futuro

Criar um dockerfile para rodar a versão antiga e verificar a equivalência com a nova versão