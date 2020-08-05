docker run \
	-it \
	--memory=64g \
	--memory-swap=64g \
	--cpuset-cpus=0-19 \
	--gpus '"device=2,3"' \
	--volume ~/summer_immersion2020:/home/user/summer_immersion2020 \
	--workdir /home/user/summer_immersion2020 \
	-p 1337:6006 \
	semenkin-immersion
