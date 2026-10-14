# Modelos do FaceLab

Os pesos **não** são versionados no Git. São baixados para esta pasta.

| Modelo | Uso | Local | Licença |
|---|---|---|---|
| `buffalo_l` (InsightFace): `det_10g.onnx` (SCRFD) + `w600k_r50.onnx` (ArcFace) + outros | Detecção, 5 landmarks, embedding de identidade | `models/models/buffalo_l/` (download automático na 1ª execução) | **Somente uso não comercial** (pesquisa/educação). O código do InsightFace é MIT, mas os pesos não. |
| (Fase 4) modelo de face swap | A definir após verificar licença e procedência | — | — |

## Regras

- Verifique a licença de qualquer modelo antes de adicioná-lo aqui.
- Não use modelos de origem desconhecida (arquivos `.onnx`/`.pth` podem ser adulterados).
- Uso comercial do FaceLab com estes pesos **não** é permitido.

