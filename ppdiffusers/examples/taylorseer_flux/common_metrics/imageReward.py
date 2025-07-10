import ImageReward as RM
model = RM.load("ImageReward-v1.0")

rewards = model.score("An image of a squirrel in Picasso style", ["/root/paddlejob/workspace/env_run/gxl/paddle_speed/ppdiffusers/examples/taylorseer_flux/text_to_image_generation-flux-dev-result.png"])
print(rewards)