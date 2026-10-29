# Modelos do FaceLab

Os pesos **não** são versionados no Git. São baixados para esta pasta via `python scripts/download_models.py` ou pelo instalador automático na inicialização do aplicativo.

| Modelo | Resolução / Uso | Local | Licença e Restrições |
|---|---|---|---|
| `buffalo_l` (InsightFace): `det_10g.onnx` (SCRFD) + `w600k_r50.onnx` (ArcFace) + `2d106det.onnx` (106 landmarks) | Detecção multi-escala (`1024×1024`), 5 e 106 marcos faciais, embedding de identidade 512-D | `models/models/buffalo_l/` | **Uso estritamente educacional / pesquisa não comercial**. O código do InsightFace é MIT, mas os pesos pré-treinados proibem exploração comercial. |
| `inswapper_128.onnx` (InsightFace) | Modelo padrão de síntese/troca facial (`128×128`, normalização RGB `[0, 1]`, projeção `emap`) | `models/inswapper_128.onnx` | **Uso estritamente não comercial / pesquisa acadêmica** (InsightFace Non-Commercial Research License). |
| `hyperswap_1a_256.onnx` (FaceFusion / HyperSwap) | Modelo opcional de troca facial em alta resolução (`256×256`, normalização RGB `[-1, 1]`) | `models/hyperswap_1a_256.onnx` | **Uso educacional e não comercial** (distribuído pelo projeto FaceFusion para fins de pesquisa). Se ausente, o FaceLab usa `inswapper_128.onnx` automaticamente como fallback. |
| `gfpgan_1.4.onnx` (Tencent ARC — GFPGAN v1.4) | Restauração e super-resolução facial (`512×512`) aplicada ao recorte alinhado do rosto | `models/gfpgan_1.4.onnx` | **Apache-2.0** (código e pesos disponibilizados pela Tencent ARC Lab; uso permitido com atribuição). |
| `bisenet_resnet_34.onnx` (BiSeNet CelebAMask-HQ) | Segmentação semântica facial de 19 classes (`512×512`, normalização ImageNet RGB) para máscara de pele e oclusão de cabelos/orelhas/óculos | `models/bisenet_resnet_34.onnx` | **MIT / Uso acadêmico** (treinado sobre CelebAMask-HQ, cujo dataset possui licença de pesquisa não comercial). |
| `xseg_2.onnx` (XSeg Occluder — opcional) | Segmentação genérica de oclusões (`256×256`) para mãos, microfones, copos e objetos em frente ao rosto | `models/xseg_2.onnx` | **Uso educacional / não comercial**. Totalmente opcional; o pipeline opera normalmente caso o arquivo não esteja presente. |

## Regras

- Verifique a licença de qualquer modelo antes de adicioná-lo aqui.
- Não use modelos de origem desconhecida (arquivos `.onnx`/`.pth` podem ser adulterados).
- Uso comercial do FaceLab com estes pesos **não** é permitido.

