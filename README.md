# Convolutional Neural Network Style 2D Convolutional

Performs a CNN style 2D Convolution with options for up-sampling and zero padding.

NOTE: As of the 9/5 due date, the most recent commit was `update names`. Since then I have used the provided solution to overcome

## Input parameters
- Input feature map data type {random, sequential}
- Input feature map size Ni input channels, Lr rows, Lc cols
- Input feature map zero padding Pl left, Pr right, Pt top, Pb bottom
- Input feature map up sampling factor Ur rows, Uc cols
- Filter coefficient data type {random, sequential}
- Filter coefficient size No output channels, Ni input channels, Fr rows, Fc cols Filter coefficient up sampling factor Dr rows, Dc cols
- Output feature map channels No output channels
- Output feature map down sampling factor Sr rows, Sc cols

## Functions

- Function: data generation Input feature maps
  - Channel x row x column Contents
  - Random
  - Sequential Filter coefficients
  - Output x input x row x col Contents
  - Random Sequential

- Function: pre processing Up sampling
  - Input feature maps
  - Filters Zero padding
  - Top, bottom, left, right

- Function: matrix creation
  - Input feature map filtering matrix Filter coefficient matrix
  - Output feature map matrix (empty)
  - Function: matrix multiplication

- Function: post processing Down sampling
  - Output feature maps

- Function: visualization Input feature maps
  - Filter coefficients Output feature maps

## Todo
- Post-Processing (Down Sampling)
- Clean up the code to use consistent parameters and simpler naming conventions
- Cleaner visualization function

## Outputs
```
Input feature matrix size (no values yet):

[[ 0.  1.  2.  5.  6.  7. 10. 11. 12.]
 [ 1.  2.  3.  6.  7.  8. 11. 12. 13.]
 [ 2.  3.  4.  7.  8.  9. 12. 13. 14.]
 [ 5.  6.  7. 10. 11. 12. 15. 16. 17.]
 [ 6.  7.  8. 11. 12. 13. 16. 17. 18.]
 [ 7.  8.  9. 12. 13. 14. 17. 18. 19.]
 [10. 11. 12. 15. 16. 17. 20. 21. 22.]
 [11. 12. 13. 16. 17. 18. 21. 22. 23.]
 [12. 13. 14. 17. 18. 19. 22. 23. 24.]
 [25. 26. 27. 30. 31. 32. 35. 36. 37.]
 [26. 27. 28. 31. 32. 33. 36. 37. 38.]
 [27. 28. 29. 32. 33. 34. 37. 38. 39.]
 [30. 31. 32. 35. 36. 37. 40. 41. 42.]
 [31. 32. 33. 36. 37. 38. 41. 42. 43.]
 [32. 33. 34. 37. 38. 39. 42. 43. 44.]
 [35. 36. 37. 40. 41. 42. 45. 46. 47.]
 [36. 37. 38. 41. 42. 43. 46. 47. 48.]
 [37. 38. 39. 42. 43. 44. 47. 48. 49.]]

Filter matrix:
[[ 0.  1.  2.  3.  4.  5.  6.  7.  8.  9. 10. 11. 12. 13. 14. 15. 16. 17.]
 [18. 19. 20. 21. 22. 23. 24. 25. 26. 27. 28. 29. 30. 31. 32. 33. 34. 35.]
 [36. 37. 38. 39. 40. 41. 42. 43. 44. 45. 46. 47. 48. 49. 50. 51. 52. 53.]]

Output matrix:

[[ 4035.  4188.  4341.  4800.  4953.  5106.  5565.  5718.  5871.]
 [10029. 10506. 10983. 12414. 12891. 13368. 14799. 15276. 15753.]
 [16023. 16824. 17625. 20028. 20829. 21630. 24033. 24834. 25635.]]
 ```
