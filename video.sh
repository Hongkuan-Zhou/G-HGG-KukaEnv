python play.py --env KukaPickAndPlaceObstacle-v1 --play_path log/520-ddpg-KukaPickAndPlaceObstacle-v1-hgg-graph-stop --play_epoch best
python play.py --env KukaPickAndPlaceObstacle-v1 --play_path log/510-ddpg-KukaPickAndPlaceObstacle-v1-hgg-stop --play_epoch best

python play.py --env KukaPickNoObstacle-v1 --play_path log/610-ddpg-KukaPickNoObstacle-v1-hgg-stop --play_epoch best
python play.py --env KukaPickNoObstacle-v1 --play_path log/620-ddpg-KukaPickNoObstacle-v1-hgg-graph-stop --play_epoch best

python play.py --env KukaPickThrow-v1 --play_path log/710-ddpg-KukaPickThrow-v1-hgg-stop --play_epoch best
python play.py --env KukaPickThrow-v1 --play_path log/720-ddpg-KukaPickThrow-v1-hgg-graph-stop --play_epoch best

python play.py --env KukaPushLabyrinth-v1 --play_path log/810-ddpg-KukaPushLabyrinth-v1-hgg-stop --play_epoch best
python play.py --env KukaPushLabyrinth-v1 --play_path log/820-ddpg-KukaPushLabyrinth-v1-hgg-graph-stop --play_epoch best