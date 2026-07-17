{- |
Module      : AuroCompute
Description : Haskell spectral / phi kernels for Auro polyglot mind
Copyright   : (c) Alfredo Medina / ItsNotAILABS, 2026
License     : Apache-2.0

JSON line CLI: pass action as args or stdin JSON.
  runghc AuroCompute.hs health
  runghc AuroCompute.hs spectral_energy 1,0,-1,0,1
  runghc AuroCompute.hs phi_powers 12
  runghc AuroCompute.hs multi_fft_embed "hello spectral"
-}

{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE StrictData #-}

module Main where

import Data.Bits (shiftR, (.&.))
import Data.Char (ord)
import Data.List (foldl')
import System.Environment (getArgs)
import Text.Printf (printf)
import Text.Read (readMaybe)

phi :: Double
phi = (1 + sqrt 5) / 2

-- | Sum of DFT magnitude bins (pure Haskell, no FFTW).
spectralEnergy :: [Double] -> Double
spectralEnergy [] = 0
spectralEnergy xs =
  let n = length xs
      half = n `div` 2
      mag k =
        let (re, im) =
              foldl'
                ( \(r, i) t ->
                    let ang = 2 * pi * fromIntegral k * fromIntegral t / fromIntegral n
                        x = xs !! t
                     in (r + x * cos ang, i - x * sin ang)
                )
                (0, 0)
                [0 .. n - 1]
         in sqrt (re * re + im * im)
   in sum [mag k | k <- [0 .. half]]

phiPowers :: Int -> [Double]
phiPowers n = [phi ** fromIntegral i | i <- [1 .. n]]

-- | Simple multi-scale energy embedding (16 dims).
multiFftEmbed :: String -> [Double]
multiFftEmbed s =
  let raw = map (fromIntegral . ord) s
      padded = if length raw < 16 then raw ++ replicate (16 - length raw) 0 else take 256 raw
      scales = 4
      bins = 4
      go i =
        let win = max 16 (round (fromIntegral (length padded) / (phi ** fromIntegral i)))
            seg = take win (padded ++ repeat 0)
            e = spectralEnergy seg
            mean = sum seg / fromIntegral (length seg)
            var = sum [(x - mean) ** 2 | x <- seg] / fromIntegral (length seg)
            peak = maximum (map abs seg ++ [0])
         in [e, mean, sqrt var, peak]
      flat = concatMap go [0 .. scales - 1]
      nrm = sqrt (sum [x * x | x <- flat]) + 1e-12
   in map (/ nrm) flat

-- | Dot product train-ish loss on small vectors.
dotTrainStep :: [Double] -> [Double] -> Double -> (Double, [Double])
dotTrainStep w x lr =
  let pred = sum (zipWith (*) w x)
      -- target = 1 for demo
      err = pred - 1.0
      loss = err * err
      grad = map (* (2 * err)) x
      w' = zipWith (\wi gi -> wi - lr * gi) w grad
   in (loss, w')

jsonList :: [Double] -> String
jsonList xs = "[" ++ joinComma (map (printf "%.10g") xs) ++ "]"
  where
    joinComma [] = ""
    joinComma [a] = a
    joinComma (a : as) = a ++ "," ++ joinComma as

parseDoubles :: String -> [Double]
parseDoubles s =
  let parts = splitOn ',' s
   in [d | p <- parts, Just d <- [readMaybe p]]

splitOn :: Char -> String -> [String]
splitOn _ [] = []
splitOn c s =
  let (a, b) = break (== c) s
   in a : case b of
        [] -> []
        (_ : rest) -> splitOn c rest

main :: IO ()
main = do
  args <- getArgs
  case args of
    [] -> putHealth
    ("health" : _) -> putHealth
    ("spectral_energy" : rest) -> do
      let xs = if null rest then [1, 0, -1, 0, 1, 0, -1] else parseDoubles (unwords rest)
          e = spectralEnergy xs
      putStrLn $
        "{\"ok\":true,\"lang\":\"haskell\",\"energy\":"
          ++ printf "%.10g" e
          ++ "}"
    ("phi_powers" : rest) -> do
      let n = case rest of
            (a : _) -> maybe 12 id (readMaybe a)
            _ -> 12
          p = phiPowers n
      putStrLn $
        "{\"ok\":true,\"lang\":\"haskell\",\"powers\":"
          ++ jsonList p
          ++ ",\"sum\":"
          ++ printf "%.10g" (sum p)
          ++ "}"
    ("multi_fft_embed" : rest) -> do
      let text = if null rest then "MESIE spectral" else unwords rest
          v = multiFftEmbed text
      putStrLn $
        "{\"ok\":true,\"lang\":\"haskell\",\"dim\":"
          ++ show (length v)
          ++ ",\"embedding\":"
          ++ jsonList v
          ++ "}"
    ("dot_train" : _) -> do
      let w = [0.1, 0.2, 0.3, 0.4]
          x = [1, 0.5, 0.25, 0.125]
          (loss, w') = dotTrainStep w x 0.01
      putStrLn $
        "{\"ok\":true,\"lang\":\"haskell\",\"loss\":"
          ++ printf "%.10g" loss
          ++ ",\"w\":"
          ++ jsonList w'
          ++ "}"
    _ ->
      putStrLn "{\"ok\":false,\"lang\":\"haskell\",\"error\":\"unknown action\"}"
  where
    putHealth =
      putStrLn
        "{\"ok\":true,\"lang\":\"haskell\",\"module\":\"AuroCompute\",\"phi\":1.6180339887,\"actions\":[\"health\",\"spectral_energy\",\"phi_powers\",\"multi_fft_embed\",\"dot_train\"]}"
