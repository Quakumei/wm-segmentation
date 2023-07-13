.PHONY: black lint clean tfds train
PYTHON=python3
TFDS_DATA_DIR=~/tensorflow_datasets

black:
	$(PYTHON) -m black .

lint: black

clean:
	rm -rf data/generated

data/generated:
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

tfds: data/generated
	tfds build wm_segmentation/tfds/WatermarkSubtitlesDatasetSegmentation.py --overwrite --data_dir $(TFDS_DATA_DIR)

train:
	$(PYTHON) -m wm_segmentation.train.train

run_tensorboard:
	tensorboard --logdir logs