---
name: comfyui-gpu
description: Crear flujos editables de vídeo con ComfyUI y ejecutarlos con GPU serverless desde el ordenador de una cuenta de Arquitectura de IA Bot.
---

Usa ComfyUI para vídeos generativos de arquitectura por texto o imagen. Para composiciones HTML con rótulos y animación exacta, sigue usando HyperFrames. Iabot 3D conserva sus herramientas de modelado y render.

En el escritorio está «ComfyUI · iabot GPU». El editor funciona en el ordenador de la empresa; los modelos y la inferencia se ejecutan en RunPod Serverless. No instales PyTorch ni descargues pesos en Daytona.

Para trabajar por terminal:

1. Consulta `comfyui-gpu health` y las plantillas en `/opt/iabot-comfyui/templates/`. El catálogo de nodos está en `/opt/iabot-comfyui/catalog.json`; corresponde a la versión instalada.
2. Copia `03_Prueba_GPU_corta_api.json` para una prueba técnica breve. Para una entrega, adapta el flujo y su calidad al encargo: las plantillas visuales de texto a vídeo e imagen a vídeo se abren en el editor. Mantén una copia editable del flujo solicitado.
3. Envía `comfyui-gpu render flujo_api.json --request-id UUID`. Usa un UUID nuevo para cada revisión, y conserva el mismo UUID si necesitas recuperar la respuesta del mismo envío. El comando devuelve enseguida el identificador del trabajo; comprueba `comfyui-gpu status UUID` sin bloquear durante minutos una llamada de terminal.
4. Cuando el estado sea `completed`, descarga con `comfyui-gpu save UUID --output ruta/video.mp4`, revisa el vídeo y adjúntalo al usuario. El resultado incluye `fileId`, `workflowFileId`, el dispositivo CUDA y los segundos reales de ejecución. Puedes adjuntar los identificadores canónicos mediante `attach_file` sin descargar otra copia al escritorio.

Para una imagen de referencia usa `--image nombre.png=ruta/imagen.png` y ese mismo `nombre.png` en LoadImage. El editor también permite subirla desde el nodo; se guarda en la biblioteca privada. No envíes rutas locales como si existieran en la GPU.

Esta primera configuración contiene Wan 2.2 TI2V 5B, el codificador de texto UMT5 y su VAE. Admite los nodos nativos del catálogo, un vídeo por trabajo, hasta 81 fotogramas (4n+1), 720p, 30 FPS y 30 pasos; las dimensiones deben ser divisibles por 16. Las imágenes de entrada y el MP4 tienen un límite de 6 MB. No instala nodos personalizados ni modelos arbitrarios desde un flujo.

El proyecto de cada ejecución, las imágenes y los vídeos permanecen en la carpeta «ComfyUI GPU» de la biblioteca de la empresa. Guarda también el flujo visual desde el editor para recuperarlo allí. El apagado del trabajador no elimina esos archivos. La página «Ver resultados» muestra cada trabajo terminado mientras otros siguen pendientes; entrega un avance útil en cuanto esté comprobado, indicándolo como provisional si queda trabajo.

Si el estado indica envío incierto, conserva su UUID y consulta su estado; no lances otra ejecución automáticamente. Una cola o un error de GPU no permiten afirmar que ya existe un vídeo ni sustituir silenciosamente la ejecución por CPU.

El servicio aplica el consumo por trabajo a la empresa que lo solicita; no hay un ordenador GPU alquilado por cliente. Mantén en Daytona solo las copias de trabajo necesarias. `iabot-archivos --help` explica cómo recuperar proyectos históricos archivados; conserva las ediciones locales.
