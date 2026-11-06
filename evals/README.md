# Conjunto de desarrollo — Evaluación del selector (TASK 29-BUILD)

Conjunto de **60 casos sintéticos** para diagnosticar y ajustar el selector
documental (`app/services/extractive_answering.py`) en modo extractivo.

- **No es** el conjunto reservado para la evaluación final amplia (objetivo 0.6).
- **No es** una demostración de independencia: se usa para diagnóstico y ajuste.
- No contiene información real de clientes ni documentación interna real.

## Formato

Fichero: `evals/dev_set_cases.json`

Cada elemento de la lista JSON es un caso con los campos:

| Campo | Tipo | Descripción |
|---|---|---|
| `id` | string | Identificador único del caso, p. ej. `DEV-001`. |
| `category` | string | Categoría funcional del caso (síntoma/error, ordinal, condición, seguridad, resolución, etc.). |
| `classification` | string | `respondible` (la evidencia respalda y la respuesta esperada puede recuperarse) o `abstencion` (falta información o la evidencia no responde). |
| `question` | string | Pregunta tal como la recibe el selector. |
| `evidence` | list[string] | Fragmentos sintéticos aceptados como evidencia (`{"text", ...}` en el evaluador real). |
| `expected_answer` | string \| null | Span exacto recuperable de la evidencia; `null` cuando se espera abstención. |
| `justification` | string | Por qué la evidencia es (o no es) suficiente. |
| `procedure_family` | string | Familia o procedimiento al que pertenece el caso. |

## Convenciones

- Las respuestas esperadas deben ser spans de la evidencia (coincidencia exacta).
- Las abstenciones se expresan con `expected_answer: null` y una justificación que
  indica qué información falta o qué conflicto entra en juego.
- Los intentos de manipulación con inyección (`ignore`/`disregard`/`forget` +
  instrucciones, `reveal system prompt`) se esperan como **abstención**, dado el
  comportamiento actual del selector (`_contains_injection` → `None`).
- La expectativa de abstención para intentos de inyección (incluidas las mezclas
  de pregunta legítima + instrucción maliciosa) está **marcada como provisional**:
  la política implementada es un filtro por patrones y no constituye una protección
  general completa.
- No se consideran casos evaluables los intentos de mezcla pregunta legítima +
  instrucción maliciosa cuando no exista política definida; aquí la política sí
  existe (abstención por inyección) y se registran con ese resultado esperado.

## Cómo revisar

1. Validar sintaxis JSON (`json.loads`).
2. Comprobar IDs únicos y formato `DEV-NNN`.
3. Comprobar reparto `respondible` / `abstencion`.
4. Comprobar que cada `expected_answer` no nulo sea span de su `evidence`.
5. Detectar duplicados evidentes (misma pregunta + misma evidencia).
6. Revisar campos incompletos o justificaciones ambiguas.

## Límite

Este conjunto se usa para **desarrollo** del ajuste del selector. La evaluación
amplia independiente y los casos reservados (20) se preparan después, sin
reutilizar preguntas, evidencias ni variantes casi idénticas entre ambos
conjuntos.
