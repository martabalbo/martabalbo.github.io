---
layout: post
title:  "Knit Pattern Writer"
date:   2026-07-03 22.19 +0100
categories: programming
---

This idea came to me when I was trying to knit my first scarf using a technique called double knitting, which produces two interlocked layers of fabric. On the scarf there was a design in two colors (and the same one on the other side, with inverted colors).

I run into a problem: when trying to follow the design, which I drew on graph paper, I had to count out how many stitches in each color before knitting every color. So, I asked myself: what if I could create something that would count for me?

Then, I put the project on hold, because I didn't have the skillset to build it.

After completing the Foundations course of [The Odin Project][the-odin-project], I realized I now had the necessary skills.

# Project overview

I wanted to build a site that allowed the user to create a grid, choosing its height and width, and fill the squares with one or more colors to create a personalized design. Then, some code would read the grid and generate a written pattern with the instructions needed to produce the design.

At first, I only built a site that allowed the user to fill a grid with only one color, which is perfect for double knitting. Then, I wanted to give the chance to use the same site for colorwork, which can use more than two colors. So, I added the possibility to select up to five colors, not including white.

Wanting to make the written pattern as clear as possible, I also added the chance for the user to rename the colors, using the user's input in the pattern.

# Building the user interface

I started by building a basic HTML page. It had a header with the title, empty spaces for the grid and the written pattern, and only three input boxes: two number input boxes for the grid height and width, and one dropdown menu for the color. At this stage, I was still thinking about double-knitting, which uses only two colors. Afterwards, I added the chance to select up to five colors via dropdown menus. I added also the "Create grid" and "Clear" buttons, to create and wipe clean the grid, and a couple of instructions to clarify what each step did.

When the grid was generating correctly, and it was possible to fill in, I added the needle selection section and the "Generate pattern" and "Print pattern" buttons. Afterwards, I added a tooltip to explain what changes the needle selection produced in the pattern.

To improve usability, I grouped all the buttons and input boxes on the left, using FlexBox to align each item. Then, I added a footer with links to the source code on GitHub.

# Creating a grid

I aleready had to create a personalized grid and change the squares' colors in the [Etch-a-Sketch][etch-a-sketch-github] exercise from [The Odin Project][the-odin-project] (you can see the end product [here][etch-a-sketch]).

I tried to use the same function, `createGrid` but it created the grid by column, which proved to be a problem when trying to parse the grid by row to generate the pattern. So, I wrote `createGridByRow`, which uses FlexBox to style the columns of the grid, making them rows, and producing a grid that was built row by row.

Before building the grid, this function clears the grid space from any previous grid. It does so by removing child nodes, if any, inside the grid div.

Then, it gets the grid dimensions chosen by the user and uses them to build the grid, appending it to the empty grid div.

# Coloring the grid

This time I needed to add a function to each square that would toggle the selected color on and off, as opposed to the [Etch-a-Sketch][etch-a-sketch-github] exercise, where the color couldn't be turned off.

To do this, I added the `toggleColor` function to the "select" button (the one that lets the user select a color after having chosen it from a dropdown menu). This function takes the selected color, and to each square adds an event listener, which changes its background from white to the color and back when clicked on.

To clear the grid, I simply used `clearGrid` to add an event listener to the "clear" button that would turn the squares' background color back to white.

# Choosing color names

The `toggleColor` function also takes the content of the text input box that contains the color name chosen by the user. It adds this name as an `id` to the square. This is the color name used by the function that generates the pattern.

# Choosing the needles

One difficulty that I encountered with my scarf project was that I needed to reverse every other row of the pattern. I had to go back and forth with each new row, which is typical when using straight needles, and so I had to start reading every second row from right to left.

To solve this issue I added the option to choose the type of needles, straight (that reverse every second row) or circular (that leave the pattern as is). I also added a tooltip to explain what changes with each choice.

# Generating the written pattern

I wrote the `generatePattern` function, that like `createGrid` removes any child nodes of the pattern div. This lets the user modify the grid and correct the pattern with the alterations.

To write this function, I started by trying to read the grid and save the information in a data structure. I chose an array for each of the rows, and decided to add each row to the pattern instead of trying to read it all in one go. Obtaining an array with one element per square, the element being a string that contained the square's color, was as easy as getting the square `id` and pushing it into the array.

Then, I had to obtain, from an array like `[white, white, blue, white]`, something like `2 white, 1 blue, 1 white`. This prove to be tricky, and I achieved by writing the `mergeArray` function. This function reads the array until the elements are the same, moves these elements into a temporary array, counts them and adds to the original array the total and the first element of the temporary array.

{% highlight javascript %}
function mergeArray(arr, mergedArr) {
    if (arr.length == 0) {
        return;
    }
    
    let temp = [];
    let i = 0;
    
    do {
        temp.push(arr.shift());
        i++;
    }
    while(temp[0]==arr[0]);

    mergedArr.push(temp.length, temp[0]);
    /*mergedArr.push(temp.reduce( //older version with prime numbers to identify colors
        (accumulator, currentValue) => accumulator * currentValue,
        0,
    ));*/
    mergeArray(arr, mergedArr);
};
{% endhighlight %}

The original array is modified in place, and can now be used to write a string containing the current row of the written pattern.

# Next steps





[the-odin-project]: https://www.theodinproject.com/
[etch-a-sketch-github]: https://github.com/martabalbo/odin-etch-a-sketch
[etch-a-sketch]: https://martabalbo.github.io/odin-etch-a-sketch/
