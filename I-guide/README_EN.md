# ReadMe

A compact notebook for disaster-image object detection and visualization with Gemini.

## What this notebook does

This notebook demonstrates a simple workflow for detecting objects in a street-view image, parsing the model output, and drawing bounding boxes on the image.

The notebook includes:

- A JSON parsing helper for removing markdown fences from model output.
- A plotting helper for drawing 2D bounding boxes and labels on images.
- Gemini client setup for running image-based generation.
- Example prompts for disaster-related object detection.
- Example calls to `gemini-2.5-pro`.
- Example code for displaying the raw response and visualizing detected objects.

## Main utilities

### `parse_json(json_output)`

This helper removes markdown code fences such as ```json blocks and returns clean JSON text.

### `plot_bounding_boxes(im, bounding_boxes)`

This helper:

- reads a PIL image,
- parses JSON-formatted detections,
- converts normalized coordinates in the range 0 to 1000 into pixel coordinates,
- draws bounding boxes and labels,
- displays the final image.

The notebook contains two plotting versions. The later version uses `ImageFont.load_default()` and is easier to run without an external font file.

## Gemini setup

The notebook shows how to:

- set `GEMINI_API_KEY`,
- create a `google.genai.Client`,

It also defines a system instruction string that tells the model to:

- return bounding boxes as JSON,
- include labels,
- avoid masks or code fencing,
- limit results to 25 objects,
- distinguish repeated objects by unique characteristics.

## Example prompt

One example prompt asks Gemini to detect disaster-related categories in a post-hurricane street-view image, including:

- damaged houses,
- destroyed buildings,
- small piles of debris,
- fallen trees or broken branches,
- telephone poles,
- power lines,
- dirt roads,
- cleared paths.

The notebook requests JSON output with these fields for each object:

- `box_2d`: `[ymin, xmin, ymax, xmax]` using normalized 0 to 1000 coordinates,
- `label`: a specific object name with category information,
- `mask_polygon`: a flat list of normalized polygon coordinates.

## Example workflow

1. Load an image with PIL.
2. Resize it with `thumbnail([1024, 1024], Image.Resampling.LANCZOS)`.
3. Send the image and prompt to `client.models.generate_content(...)`.
4. Display the response text with `Markdown(response.text)`.
5. Render the detections with `plot_bounding_boxes(im, response.text)`.

## Output format

The notebook is designed around strict JSON output. If the model returns fenced JSON, the parsing helper removes the fencing before visualization.

## Notes

- The notebook uses `gemini-2.5-pro` in the example calls.
- The image path shown in the notebook points to a dataset file under `/Dataset/SVI & RSI_CVIAN_selected_50/...`.
- Some inline comments in the notebook are in Chinese, but the overall workflow is straightforward: load image, generate detections, parse JSON, and draw boxes.

## Minimal dependencies

Based on the notebook, the main packages used are:

- `Pillow`
- `google-genai`
- `IPython`