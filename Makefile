#!/bin/bash
local:
	docker build -t wavespectra2dsplitfit:deploy .
	docker run -it -p 8888:8888 -v $(shell pwd):/home/shell/wavespectra2d wavespectra2dsplitfit:deploy jupyter lab --port=8888 --ip=0.0.0.0


