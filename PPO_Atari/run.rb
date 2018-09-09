
epochs = [4, 8, 16]
batchsize = [32, 64, 128, 256, 512]
timesteps =[256, 512, 1024]
learning_rate = ["0.001", "0.0001", "0.00001", "0.000001","0.0000001"]
File.open("run.bat","w") do |f|
	for e in epochs do 
		for l in learning_rate do
			for t in timesteps do
				for b in batchsize do
					if b <= t 
						str = "python  PPO_Atari.py --epoch %d -b %d -t %d  -lr %s" % [e , b,  t , l]
						f.puts str
					end
				end
			end
		end
	end
end