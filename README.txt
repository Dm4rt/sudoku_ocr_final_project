# CS493-HW3

## To compile: make all
Alternatively: make h1, make h2, make h3, make h4

--> Add information based on your programs

I completed all h1, h2, h3, and h4

For h1 i encountered no bugs. It was just getting an output similar to the example outputs. Then I realized after talking to classmates that it doesn't need to be exact because everyone is using different masks and algorithms. The only thing I changed was square rooting the mask, so instead of just having Gx^2 + Gy^2, I square rooting the results, since that's what the professor mentioned in the class. Additionally, I applied the squared gradient  mask again and got a more defined results.

to run h1 it's just ./h1 {input_image_name} {output_edge_image_name}

h2 was no issue, I just reused the code from homework 1 as instructed. The threshold I tested with is 50 and 30 because I tried to get the defined image like the example the professor gave.
 
To use h2, it's just ./h2 {input_image_name} {threshold} {output_image_name}


for H3 there were a few bugs, mainly segmentation fault errors that had to do with how I. was implementing rho and theta. Additionally, like the last assignment for some reason the output was like rotated and flipped weirdly, so I had to fix that. Also issues with the voting and normalization so I had to incorrect max-vote scaling leading to me having to properly calculate the max_votes for grayscale output. Also the way I was writing to the output array file. There was misaligned with how I was writing the matrix, so I had to properly format the row and column output for the output array. In the end I stuck to outputting a file that's expected to be a txt file. The txt file just stores the matrix/ hough array.

to use h3: Usage: ./h3 {input_binary_image_name} {output_hough_image_name} {output_hough_array_name}

for H4, there were a lot of bugs because it was the biggest and probably the most difficult program in this homework for me. There were some difficulties at first with how I was reading the accumulator array due to the file that I was reading from h3. Additionally, there was some confusion between how the image class and the logic in my algorithm coordinated the row and column ordering. Also there were issues with how I was mapping the hough parameters like rho and theta to the line equations. On top of that the actual line detect produced some skewed lines that were all over the place and incorrectly position, but this was all fixed after debugging in class and the tips the professor gave. I had to filter the lines a bit but overall, it was a lot of back and for the with debugging and seeing which of these errors was causing segmentation errors and where. The input hough array is expecting a txt array file. It iterates through the txt file to count the amount of lines to get the dimensions.

to use h4 Usage: ./h4 input_image input_hough_array input_threshold output_line_image

The input image I used to test all the programs in this sequence of programs is hough_simple_1.pgm. In other words, I used hough_simple_1.pgm first, and then I used the output for each subsequent program. So the output from h1 I used for h2, and the output for h2 I used for h3 and then the output of h3 for h4.