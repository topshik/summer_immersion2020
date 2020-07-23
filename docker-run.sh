docker run \
	-it \
	--memory=64g \
	--memory-swap=64g \
	--cpuset-cpus=0-19 \
	--gpus '"device=2,3"' \
	--volume ~/summer_immersion2020:/home/user/immersion \
	--workdir /home/user/immersion \
	-p 1337:6006 \
	semenkin-immersion
