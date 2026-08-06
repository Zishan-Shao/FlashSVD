   (Note: I will ensure the markdown structure is identical, only replacing English text with Spanish.)
   Done. 
   *Self-Correction/Note during generation prep*: I notice the prompt says "Preserve Markdown, code blocks, commands, identifiers, image targets, and link targets exactly." This means I should not modify the code blocks at all. I will copy-paste them exactly. The rest gets translated.
   All set. 
   Output matches the response. 
   [Proceeds]
</think>

# FlashSVD v1.5: Haciendo que la inferencia de Transformers de bajo rango sea realmente rápida

FlashSVD es un entorno de ejecución de inferencia en streaming para modelos de lenguaje comprimidos con SVD.
Este repositorio ahora trata **FlashSVD v1.5** como la ruta principal del producto.

El código principal del entorno de ejecución permanece en los directorios de nivel superior existentes como
`models/`, `runtime/`, y `utils/`. El punto de entrada de la demostración ligera v1.5
es [demo_flashsvd_v15.py](demo_flashsvd_v15.py), con el código auxiliar bajo
`scripts/demo_support/`.

![FlashSVD v1.5 Pipeline](docs/figures/FlashSVD-V1.5-Pipeline.png)

Enlaces rápidos:

- Demo: [ejecución en un solo comando](#demo)
- Estado: [dirección de servicio actual](docs/notes/CURRENT_STATUS.md)
- Estructura: [notas de arquitectura](docs/architecture.md)
- Benchmarks: [punto de entrada del benchmark de decodificación](benchmark/decode/bench_flashsvd_vs_svd_decode.py)

## Estructura Principal

```text
demo_flashsvd_v15.py  root-level v1.5 demo entrypoint
models/               Hugging Face-facing model integration
runtime/              decode dispatch, cache plumbing, backend selection
kernels/              Triton / kernel implementations
src/                  compression and conversion flows such as SVD-LLM
utils/                loading, evaluation, checkpoint utilities
benchmark/            benchmark and correctness scripts for v1.5
scripts/              demo support helpers, job scripts, and smoke tests
docs/                 repository architecture notes
```

Más detalles: [docs/architecture.md](docs/architecture.md)

## Dirección Actual de Servicio

La ruta de decodificación en producción sigue el entorno de ejecución v1.5 descrito en
[docs/notes/CURRENT_STATUS.md](docs/notes/CURRENT_STATUS.md):

- atención: caché KV denso + reconstrucción del token actual + `flash_attn_with_kvcache`
- MLP: `auto` con enrutamiento habilitado para gráficos hacia la ruta de producción compactada

Las notas verificadas en ese archivo se revisaron por última vez el 10 y 11 de marzo de 2026 y
reportan un aumento de velocidad de decodificación de extremo a extremo de aproximadamente `1.50x` a `1.53x` en la configuración
documentada de Llama-2-7B.

## Instalación

FlashSVD v1.5 requiere Python `3.10+` y un entorno CUDA/PyTorch si deseas
ejecutar la ruta fusionada de GPU.

Configuración inicial recomendada del entorno:

```bash
git clone https://github.com/Zishan-Shao/FlashSVD.git
cd FlashSVD
conda env create -f environment.yml
conda activate flashsvdv15
pip install -e .[test]
```

Si ya gestionas tu propio entorno de CUDA, las dependencias del repositorio son:

```bash
# install the pinned runtime stack used by this repo
pip install -r requirements.txt

# optional: install repo extras such as pytest
pip install -e .[test]
```

Los asistentes de demostración son locales al repositorio y residen bajo `scripts/demo_support/`; no
requieren una distribución de paquetes de nivel superior separada. Si necesitas una versión diferente
de PyTorch/FlashAttention, instálala primero y luego instala las
dependencias del repositorio.

## Inicio Rápido

### API de Python

```python
from scripts.demo_support import (
    cast_model_for_inference,
    configure_runtime,
    generate_text,
    get_model_from_source,
)

configure_runtime(
    mode="flashsvd",
    ffn_backend="auto",
    enable_mlp_graph=True,
    mlp_graph_scope="layer_tail",
    enable_flash_dense_attn=True,
)

model, tokenizer = get_model_from_source("/path/to/checkpoint-or-hf-export")
model = cast_model_for_inference(model, dtype="auto", device="cuda")

result = generate_text(
    model,
    tokenizer,
    prompt="FlashSVD accelerates low-rank language models by",
    max_new_tokens=64,
    device="cuda",
)
print(result["completion_text"])
```

### Demo

Desde la raíz del repositorio, el siguiente comando para copiar y pegar ejecuta el punto de control de demostración público de
FlashSVD v1.5 de extremo a extremo:

```bash
CUDA_VISIBLE_DEVICES=0 python demo_flashsvd_v15.py \
  --checkpoint Duke-CEI-SVD/LowRankArena::llama_7b/Basis_Sharing/share_llama-7b_20 \
  --device cuda \
  --dtype auto \
  --prompt "Explain in one sentence: FlashSVD accelerates low-rank language models by" \
  --max-new-tokens 24 \
  --warmup-tokens 16
```

Si ya tienes el entorno `flashsvdv15` del paso de instalación
activo, ese comando debería ejecutarse directamente sin necesidad de preparación
local del punto de control. Carga la exportación pública de LLaMA-7B `Basis Sharing 0.8` de LowRankArena,
que no es el ejemplo público más rápido, pero es un punto de control de demostración
mucho mejor para una primera ejecución que la exportación `SVD-LLM v1 update 0.5`.

Esto configura la receta actual de servicio de FlashSVD v1.5:

- decodificación dense-KV habilitada
- backend FFN `auto`
- gráfico CUDA MLP habilitado con ámbito `layer_tail`

La demo ahora imprime tanto la velocidad `cold` como la velocidad `steady_state`. La primera
incluye costos únicos como el autotuning de decodificación densa y la captura de gráficos CUDA;
la segunda vuelve a ejecutar el mismo prompt una vez llenas esas cachés y es el
valor más adecuado para una verificación rápida. Puedes añadir `--warmup-tokens N` si deseas
una pasada de calentamiento sin cronometrar entre estas dos mediciones.

Si deseas comparar con la ruta de respaldo utilizando el mismo prompt y conteo de
tokens, ejecuta el mismo comando con `--mode hf`.

### Benchmarks

Benchmark principal de decodificación:

```bash
python benchmark/decode/bench_flashsvd_vs_svd_decode.py \
  --checkpoint /path/to/checkpoint.pt \
  --dtype bf16 \
  --device cuda \
  --prompt_len 512 \
  --new_tokens 32 \
  --warmup 3 \
  --batch_size 1 \
  --flashsvd_ffn_backend auto \
  --experimental_flash_dense_attn \
  --mlp_graph \
  --mlp_graph_scope layer_tail \
  --baseline_dense_kvcache
```

Verificación de corrección:

```bash
python benchmark/decode/check_flashsvd_decode_correctness.py \
  --checkpoint /path/to/checkpoint.pt \
  --dtype bf16 \
  --device cuda \
  --batch_size 1 \
  --decode_steps 16 \
  --legacy_backend flashsvd_mlp_dual_split_exact_legacy \
  --test_backend flashsvd_mlp_dual_split_prod \
  --flash_dense_attn \
  --baseline_dense_kvcache \
  --reference_dense_attn
```

## Compatibilidad

Todavía se admiten importaciones legadas como `component.*`, `flashsvd_component.*`, `models.*`,
`runtime.*`, y `utils.*` para que los puntos de control locales y los scripts más
antiguos sigan cargándose.
