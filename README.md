# CPU distributed image segmentation

This repository implements the image segmentation tutorial from tensorflow (https://www.tensorflow.org/tutorials/images/segmentation) within a CPU multi-worker distributed training.

The application is dockerised within a single multi-purpose image, and then deployed into a K8s cluster.
The multiple worker deployments, together with the corresponding services, are synchronised with the help of K8s-yamlfied classes (https://github.com/paoloalba/k8s_deployer).
The programmaticly generate yaml files can be normally deployed by ```kubectl``` inline commands or any preferred script.