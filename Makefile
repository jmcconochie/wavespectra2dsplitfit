#!/bin/bash
local:
	docker image prune -f
	docker build -t wavespectra2dsplitfit:deploy .
	docker image prune -f
	#docker run -it -v $(shell pwd):/home/shell/wavespectra2d wavespectra2d-splitfit:deploy /bin/bash
	docker run -it -p 8888:8888 -v $(shell pwd):/home/shell/wavespectra2d wavespectra2dsplitfit:deploy jupyter lab --port=8888 --ip=0.0.0.0


