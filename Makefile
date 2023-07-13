.PHONY: black lint clean
PYTHON=python3

black:
	$(PYTHON) -m black .

lint: black

clean:
	rm -rf data/generated

data/generated: clean
	mkdir -p data/generated
	$(PYTHON) -m wm_segmentation.data.cli \
			data/source/unwatermarked_images/train/ \
			data/source/watermarks/ \
			data/source/subtitles_phrases.txt \
			$@/train
	$(PYTHON) -m wm_segmentation.data.cli \
			data/source/unwatermarked_images/validate/ \
			data/source/watermarks/ \
			data/source/subtitles_phrases.txt \
			$@/validate
