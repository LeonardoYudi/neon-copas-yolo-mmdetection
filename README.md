# Detecção de copas de árvores no NeonTreeEvaluation: YOLO × MMDetection

Este repositório documenta como foram treinados e avaliados os 11 detectores comparados no Trabalho de Conclusão de Curso: dados, recortes, versões das bibliotecas, parâmetros de cada modelo, critério de seleção e protocolo de avaliação.

> **Pesos e arquivos de apoio:** os pesos treinados, os `config.py` e `args.yaml` de cada modelo, as métricas por semente (`metricas.csv`) e as listas do dataset serão adicionados a este repositório em breve. As seções [Conteúdo do pacote](#conteúdo-do-pacote) e [Como usar](#como-usar) já descrevem esses arquivos.

São 22 modelos: 11 arquiteturas, cada uma treinada com duas sementes (0 e 1). Todos foram treinados no conjunto **neon-400-sem-sobreposicao**, por **60 épocas** e com entrada de **512 px**.

O material serve para três usos:

- **Testar** os modelos prontos (seção [Como usar](#como-usar)).
- **Conferir** os números do trabalho (seção [Resultados](#resultados)).
- **Replicar** o experimento do zero (seção [Como replicar](#como-replicar)).

---

## Resultados

Conjunto de teste do NeonTreeEvaluation: 194 imagens RGB de 400 × 400 px, com 6.633 copas anotadas. A tabela mostra a média das sementes 0 e 1. O protocolo de medição está detalhado em [Protocolo de avaliação](#6-protocolo-de-avaliação).

| Modelo | Framework | Parâmetros (M) | P | R | F1 | AP50 | AP50:95 | R_max |
|---|---|---|---|---|---|---|---|---|
| Faster R-CNN R50-FPN | MMDetection | 41,3 | 0,730 | 0,674 | **0,700** | 0,534 | 0,207 | 0,760 |
| ATSS R50-FPN | MMDetection | 32,1 | 0,737 | 0,660 | 0,696 | **0,564** | **0,215** | 0,833 |
| RetinaNet R50-FPN | MMDetection | 36,3 | 0,746 | 0,635 | 0,686 | 0,489 | 0,197 | 0,666 |
| DETR R50 | MMDetection | 41,6 | 0,718 | 0,637 | 0,675 | 0,354 | 0,116 | 0,662 |
| SSD512 VGG16 | MMDetection | 24,4 | 0,743 | 0,607 | 0,668 | 0,524 | 0,192 | **0,856** |
| YOLOv8-S | Ultralytics | 11,1 | 0,691 | 0,630 | 0,659 | 0,518 | 0,193 | 0,829 |
| YOLOv8-M | Ultralytics | 25,9 | 0,654 | 0,648 | 0,651 | 0,506 | 0,188 | 0,801 |
| YOLOv8-N | Ultralytics | 3,0 | 0,671 | 0,631 | 0,650 | 0,519 | 0,194 | 0,847 |
| YOLO26-M | Ultralytics | 21,8 | 0,681 | 0,591 | 0,632 | 0,503 | 0,199 | 0,808 |
| YOLO26-N | Ultralytics | 2,5 | 0,702 | 0,542 | 0,612 | 0,462 | 0,183 | 0,788 |
| YOLO26-S | Ultralytics | 9,9 | 0,598 | 0,625 | 0,610 | 0,449 | 0,176 | 0,738 |

Para comparação, o LACE (Pitts, Li e Liang, arXiv 2609.26549) relata F1 de 0,728 no mesmo protocolo.

O arquivo `metricas.csv` traz os valores de **cada semente**, junto com o limiar de confiança, a época escolhida e a latência por imagem.

---

## Conteúdo do pacote

```
modelos/<modelo>/semente<0|1>/
    <modelo>_s<n>.pth | .pt   pesos em fp32, idênticos aos do treino
    config.py                 config MMDetection completo, que é a receita de treino (só nos modelos MMDetection)
    args.yaml                 argumentos de treino do Ultralytics, que são a receita (só nos YOLO)
    meta.json                 épocas, época escolhida, lote, entrada, semente, tempo de treino
dataset/
    recorte.json              parâmetros com que os recortes foram gerados
    lista_treino.txt          os 1.697 recortes de treino, por nome
    lista_validacao.txt       os 233 recortes de validação
    lista_teste.txt           as 194 imagens de teste
    data.yaml                 exemplo de data.yaml para o Ultralytics
metricas.csv                  métricas por modelo e semente (separador ;)
SHA256SUMS.txt                hash de cada arquivo
```

Os pesos dos modelos MMDetection guardam só os tensores e a lista de classes. O estado do otimizador e o histórico do treino foram retirados. Nos YOLO, os caminhos absolutos da máquina de treino foram trocados por caminhos relativos.

---

## Como usar

### YOLOv8 e YOLO26 (Ultralytics)

O YOLO26 exige `ultralytics>=8.4`. Os pesos foram treinados na versão 8.4.96.

```python
from ultralytics import YOLO

modelo = YOLO("modelos/yolov8s/semente0/yolov8s_s0.pt")
resultado = modelo.predict("imagem.png", imgsz=512, conf=0.095, max_det=300)  # limiar de metricas.csv
caixas = resultado[0].boxes.xyxy
```

### Faster R-CNN, ATSS, RetinaNet, SSD512 e DETR (MMDetection)

```python
from mmdet.apis import init_detector, inference_detector

# Só para o SSD512: corrige um erro do mmdet 3.3.0. Sem esta linha, a predição falha com
# "'SSDHead' object has no attribute 'loss_cls'".
from mmdet.models.dense_heads.ssd_head import SSDHead
SSDHead.loss_cls = None

modelo = init_detector("modelos/atss/semente0/config.py",
                       "modelos/atss/semente0/atss_s0.pth", device="cuda:0")
pred = inference_detector(modelo, "imagem.png").pred_instances
caixas = pred.bboxes[pred.scores >= 0.195]          # limiar de metricas.csv
```

**O limiar muda de modelo para modelo.** Cada execução tem o seu, escolhido na validação, e para reproduzir o F1 é preciso usar o valor de `metricas.csv`.

---

## Como replicar

Esta seção descreve os critérios usados para gerar cada modelo. Seguindo-a, dá para treinar os 22 modelos do zero e chegar aos mesmos resultados, dentro da variação esperada (item 7).

### 1. Dados de origem

O trabalho usa o benchmark **NeonTreeEvaluation** (Weinstein et al.), apenas na modalidade RGB, com resolução de 0,1 m/px.

> Weinstein, B. G. et al. A benchmark dataset for canopy crown detection and delineation in co-registered airborne RGB, LiDAR and hyperspectral imagery from the National Ecological Observation Network. *PLoS Computational Biology*, 17(7): e1009180, 2021. https://doi.org/10.1371/journal.pcbi.1009180
>
> Dados e ferramenta de avaliação: https://github.com/weecology/NeonTreeEvaluation

| Uso | O que é | Quantidade |
|---|---|---|
| Treino e validação | tiles de treino RGB do benchmark, com anotação manual das copas | 20 tiles, 23.465 caixas, 14 sítios |
| Teste | as imagens oficiais de avaliação do benchmark, usadas inteiras, sem recorte | 194 imagens de 400 × 400 px, 6.633 caixas, 22 sítios |

As outras modalidades do pacote do NEON (hiperespectral, LiDAR e CHM) **não** foram usadas.

### 2. Geração dos recortes de treino e validação

Os parâmetros estão em `dataset/recorte.json`. As listas em `dataset/` permitem conferir, nome a nome, se o recorte saiu igual.

| Critério | Valor | Por quê |
|---|---|---|
| Tamanho da janela | 400 × 400 px | O mesmo tamanho das imagens de teste: as copas ficam com o mesmo tamanho em pixels no treino e no teste. |
| Sobreposição entre janelas | nenhuma | Nenhuma copa aparece em dois recortes. |
| Faixa de borda que não completa uma janela | descartada | Recuar a última janela até a borda repetiria parte da janela vizinha e duplicaria copas. |
| Caixa cortada pela janela | é aparada no limite da janela e só entra se ≥ 50% da área original ficar dentro | O modelo não deve aprender a detectar pedaços de copa. |
| Tiles com anotação parcial | descartados quando têm menos de 0,5 caixa por janela | Dois tiles foram descartados: `2019_SJER_4_251000_4103000` (0,064 caixa por janela, 40 caixas) e `2019_TOOL_3_403000_7617000` (1 caixa). Sobram 18 tiles de 13 sítios. |
| Janelas sem copa | mantidas, até 0,3 × o número de janelas com copa de cada tile, escolhidas a intervalos regulares | Ensinam o modelo a recusar o fundo sem dominar o conjunto. |
| Divisão entre treino e validação | a cada 8 recortes de cada tile, 1 vai para a validação | É o esquema do artigo de referência. Treino e validação vêm dos mesmos tiles, e a validação serve só para escolher a época e o limiar. |

Resultado: **1.697 recortes de treino e 233 de validação**, com 21.062 caixas no total. A seleção não envolve sorteio, então os mesmos parâmetros geram sempre os mesmos recortes.

### 3. Critérios comuns a todos os modelos

| Critério | Valor |
|---|---|
| Pesos iniciais | detector completo pré-treinado no COCO: checkpoints oficiais da OpenMMLab no MMDetection e `yolov8{n,s,m}.pt` e `yolo26{n,s,m}.pt` oficiais no Ultralytics |
| Classes | 1 (`Tree`). A cabeça de classificação é reiniciada e o restante da rede vem do COCO. |
| Épocas | 60, sem parada antecipada |
| Entrada | 512 px. O recorte de 400 px é ampliado 1,28×, mantendo a proporção. |
| Lote | 4 |
| Precisão | mista (AMP) |
| Aumento de dados | o mesmo nas duas bibliotecas: escala aleatória de 0,9 a 1,1 e espelhamento horizontal com probabilidade 0,5. Sem mosaico, sem mudança de cor e sem rotação. |
| Sementes | 0 e 1, com um treino independente para cada |
| Receita | a receita **oficial** de cada biblioteca para cada arquitetura, sem ajuste fino de hiperparâmetros. A única adaptação é a agenda de taxa de aprendizado, reescalada para 60 épocas. |
| Validação | ao fim de cada época |

### 4. Critérios específicos de cada modelo

A receita completa está no `config.py` e no `args.yaml` de cada pasta. A tabela resume o que muda entre os modelos.

| Modelo | Otimizador | lr | Momento / weight decay | Agenda | Aquecimento |
|---|---|---|---|---|---|
| Faster R-CNN | SGD | 0,02 | 0,9 / 1e-4 | lr × 0,1 nas épocas 40 e 55 | linear, 500 iterações a partir de 0,001 × lr |
| ATSS | SGD | 0,01 | 0,9 / 1e-4 | idem | idem |
| RetinaNet | SGD | 0,01 | 0,9 / 1e-4 | idem | idem |
| SSD512 | SGD | 0,002 | 0,9 / 5e-4 | idem | idem |
| DETR | AdamW | 1e-4 (backbone × 0,1) | — / 1e-4; gradiente cortado em norma 0,1 | lr × 0,1 na época 40 | nenhum |
| YOLOv8 e YOLO26 (N, S e M) | AdamW (`optimizer=auto`) | 0,002 | 0,9 / 5e-4 | decaimento linear até 0,01 × lr | 3 épocas |

**Detalhes que mudam o resultado**

- **Agenda reescalada.** As receitas oficiais do MMDetection são de 12 épocas (*1x*) ou de 150 (DETR). Os marcos foram multiplicados por 60/12 e por 60/150, respectivamente.
- **DETR com 100 consultas,** o mesmo número do checkpoint COCO. Assim as consultas pré-treinadas são carregadas. Com 300, o tensor muda de forma e as consultas começam do zero, sem nenhum aviso no log. O custo de manter 100 é que o DETR emite no máximo 100 caixas por imagem, e 12 das 194 imagens de teste têm mais copas que isso.
- **SSD512:** o *padding* da imagem foi arredondado para múltiplos de 32, como nos outros modelos. Com o valor original (1), cada lote saía com um tamanho diferente e a memória da GPU de 8 GB se fragmentava. A configuração do SSD que repete o dataset várias vezes por época (`RepeatDataset`) foi retirada, para que uma época signifique o mesmo em todos os modelos.
- **YOLO com `optimizer=auto`.** O `args.yaml` mostra `lr0: 0.01`, mas no modo `auto` o Ultralytics descarta esse valor e usa AdamW com lr 0,002, porque o treino tem menos de 10 mil iterações. O `lr` registrado no `results.csv` do treino confirma o 0,002. Os campos `auto_augment` e `erasing` do `args.yaml` são de classificação e não têm efeito em detecção.

### 5. Critério de seleção: época e limiar

Nada é escolhido pelo conjunto de teste.

1. **Época:** a de **maior AP50:95 na validação**, calculado pelo próprio MMDetection ou Ultralytics ao fim de cada época. Ela está no campo `best_epoch` do `meta.json`, e os pesos publicados são os dessa época.
2. **Limiar de confiança:** a regra do platô. Na validação, a IoU 0,4, calcula-se o F1 somando acertos e erros de todas as imagens, em cada limiar de 0,005 a 0,95 (passos de 0,005). O limiar escolhido é o **menor cujo F1 fica a até 0,01 do máximo**.
   - **Por que não o limiar de F1 máximo:** a validação é feita de recortes dos tiles de treino, e o teste são as imagens do benchmark, com outra densidade de copas e outros anotadores.
   - **O ganho medido:** sob essa diferença, o limiar de F1 máximo perdeu em média 0,049 de F1 no teste, e a borda baixa do platô perdeu 0,017.
3. **Teste:** mede uma única vez, com a época e o limiar já escolhidos.

### 6. Protocolo de avaliação

| Item | Valor |
|---|---|
| Casamento | guloso por confiança: cada copa anotada casa com no máximo uma predição, com IoU ≥ 0,4 |
| P, R e F1 | **média por imagem** (macro), como no artigo de referência. Uma imagem com 3 copas pesa o mesmo que uma com 150. |
| R_max | a maior revocação ao varrer o limiar: o teto de copas que o modelo consegue achar |
| AP50 e AP50:95 | pycocotools, com até 300 detecções por imagem |
| Predição | até 300 caixas por imagem, confiança mínima de 0,001 e NMS com IoU de 0,5. O DETR emite até 100 (item 4). |
| Latência | média por imagem, uma imagem por vez, depois de 10 aquecimentos, com a GPU sincronizada. Inclui leitura do disco, pré e pós-processamento e NMS. |

As duas bibliotecas rodam em ambientes diferentes (abaixo), então parte da diferença de latência entre YOLO e MMDetection vem da pilha de software, não da arquitetura.

### 7. Ambiente e o que esperar ao replicar

| | MMDetection | Ultralytics |
|---|---|---|
| Python | 3.11.9 | 3.12.4 |
| PyTorch | 2.1.2 + CUDA 12.1 | 2.5.1 + CUDA 12.1 |
| Bibliotecas | mmdet 3.3.0, mmcv 2.1.0, mmengine 0.10.4 | ultralytics 8.4.96 |
| GPU | 8 GB de VRAM | idem |

Tempo de treino na GPU acima, por semente, segundo o campo `train_time_s` do `meta.json`:

- cerca de 0,9 a 1,6 h por modelo MMDetection;
- cerca de 0,6 a 1,2 h por YOLO;
- **cerca de 12 h para os 11 modelos**, ou seja, cerca de 25 h para as duas sementes.

**Os números não saem idênticos bit a bit.** O treino em GPU usa operações não determinísticas (`deterministic=False` no MMDetection), então refazer com a mesma semente não reproduz exatamente os mesmos pesos. Mesmo assim, o F1 deve cair dentro da variação medida entre as nossas duas sementes, de **±0,002 a ±0,009** conforme o modelo. Para conferir os números publicados sem treinar de novo, use os pesos deste pacote: são os mesmos tensores que geraram a tabela.

---

## Licenças

- **Pesos dos YOLO:** treinados com o Ultralytics, distribuídos sob **AGPL-3.0**.
- **Pesos e configs do MMDetection:** o MMDetection e os checkpoints da OpenMMLab são **Apache-2.0**.
- **Dados:** valem os termos do NeonTreeEvaluation. As imagens não estão neste repositório.

## Verificação dos arquivos

Quando os pesos forem adicionados, o arquivo `SHA256SUMS.txt` permitirá conferir a integridade de cada um:

```bash
sha256sum -c SHA256SUMS.txt
```
