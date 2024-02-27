# This file is here just to define the TwoCriticsPolicy for PPO-Lagrangian
from stable_baselines3.common.policies import (ActorTwoCriticsPolicy,
                                               ActorTwoCriticsCnnPolicy,
                                               DistributionalActorTwoCriticsPolicy,
                                               register_policy)

TwoCriticsMlpPolicy = ActorTwoCriticsPolicy
DistributionalTwoCriticsMlpPolicy = DistributionalActorTwoCriticsPolicy

register_policy("TwoCriticsMlpPolicy", ActorTwoCriticsPolicy)
register_policy("TwoCriticsCnnPolicy", ActorTwoCriticsCnnPolicy)
register_policy("DistributionalTwoCriticsMlpPolicy", DistributionalActorTwoCriticsPolicy)
