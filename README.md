gppe
====

Gaussian Process Preference Elicitation (GPPE) package. 
Implements the preference elicitation model of Bonilla et al [1].

Author: 
-----
Edwin V. Bonilla
Last update: 22/05/2012


Main Contents
-----------
1. elicit_gppe.m: Elicit preferences for a new user with the GPPE model.
2. learn_gppe.m:  Computes the negative marginal likelihood and its gradients wrt to the hyper-paramters. This can be used to learn a gppe model with Carl Rasmussen's minimize function
3. toy_example.m: A toy example demonstrating the use of the gppe package


References
-----------
[1] Edwin V. Bonilla,  Shengbo Guo and Scott Sanner.
Gaussian process preference elicitation.
In Advances in Neural Information Processing Systems 23: NIPS'10.
