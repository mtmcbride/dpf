library(testthat)
library(dpf)

test_that("resampleSubOptimal: unique weights", {
    s = resampleSubOptimal(1:500, 5)
    expect_equal(sum(s != 0), 5)
    expect_equal(matrix(tail(s, 5), ncol = 1), matrix(496:500/sum(496:500), ncol = 1))
    expect_equal(s[1:495], rep(0, 495))
    remove(s)
    s = resampleSubOptimal(30:1, 5)
    expect_equal(sum(s != 0), 5)
    expect_equal(matrix(head(s, 5), ncol = 1), matrix(30:26/sum(30:26), ncol = 1))
    expect_equal(s[6:30], rep(0, 25))
})

test_that("resampleSubOptimal: same weights", {
    s = resampleSubOptimal(c(1:24, rep(25, 3), 28:30), 5)
    expect_equal(sum(s != 0), 5)
    expect_equal(s[28:30], 28:30/(50 + sum(28:30)))
    expect_equal(s[1:24], rep(0, 24))
})