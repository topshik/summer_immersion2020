docker run \
	-it \
	--memory=64g \
	--memory-swap=64g \
	--cpuset-cpus=0-9 \
	--gpus device=2 \
	--volume ~/summer_immersion2020:/home/user/immersion \
	--workdir /home/user/immersion \
	-p 6006:6006 \
	semenkin-immersion

