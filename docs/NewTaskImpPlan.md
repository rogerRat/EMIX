针对以下12个任务：
blocks ranking rgb 
blocks ranking size
Hanging Mug
Place Bread Basket
Place Bread Skillet
Place Burger Fries
Place Can Basket
Put Cans Plasticbox
Place Dual Shoes
Stack Blocks Two
Stack Bowls Two
Scan Object-<候选>
实现以下要求：
参考当前的place_object_scale任务，全部（1）转换为全部使用固定左臂进行，严格排除右臂在数据采集和模型评测的时候的任何举动，并且严格保证在实际上模型的评估的时候右臂固定被传入的参数固定，不对当前的策略造成影响。（2）相机视角全部转换为left_oppo_cam版本，参考当前的place_object_scale_left_oppo_cam脚本的实现，对其进行镜像和复用。（3）物体的随机化的范围进行扩大，保证机器人的行进轨迹之间有足够多的交叉，参考place_object_scale_left_oppo_cam和原版的RoboTwin原生任务place_object_scale_left的区别。
开发内容：1.env下的脚本。2.对应的任务配置文件。3.对应的可能需要兼容的instructions。4.其他的可能需要的兼容性修改。5.评估步数限制在和改造之前的原任务一致。
验收结果：十二个开发完毕的任务，以及他们对应的收集到的测试数据集：一个测试数据集经过RoboTwin收集，然后转换为Lerobot格式数据。转换固定使用84x84图像大小。测试数据集只收集十个ep。同时，报告收集过程中的seed成功率，可知随机化程度和成功率往往成负相关关系。尽可能保持成功率不低于20%.