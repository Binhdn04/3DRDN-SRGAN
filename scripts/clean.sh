#!/bin/bash
rm training.log
rm results/plots/training/*
rm results/plots/validation/*
rm results/generated_images/*
rm checkpoints/tensor_checkpoints/generator_checkpoint/G/*
rm checkpoints/tensor_checkpoints/generator_checkpoint/F/*
rm checkpoints/tensor_checkpoints/discriminator_checkpoint/Y/*
rm checkpoints/tensor_checkpoints/discriminator_checkpoint/X/*
