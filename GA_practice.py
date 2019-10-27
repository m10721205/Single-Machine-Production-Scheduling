import numpy as np
import random
jobs = 5 #5, 10, 20, 50, 100 >> DNA的size
crossRate = 0.8 #交配機率
mutateRate = 0.003#突變機率
popSize=50 #人口數
iterations = 100#每種jobs跑5次
class GA(object):
    def __init__(self, DNAsize, cross_rate, mutate_rate, pop_size):
        self.DNAsize = DNAsize
        self.cross_rate = cross_rate
        self.mutate_rate = mutate_rate
        self.pop_size = pop_size
        self.pop = np.vstack([np.random.permutation(DNAsize) for _ in range(pop_size)])

    def translateDNA(self, DNA, jobs_position):
        xx = np.empty_like(DNA, dtype=np.float32)  #x座標的空矩陣
        for i, d in enumerate(DNA):              #列舉DNA中每個job編號
            jobs_coord = jobs_position[d]        #把隨機產生的job_position(Pi值)存入job_coord
            xx[i, :] = jobs_coord[:, 0]          #將job_coord中的第0個column(直的)值存入xx陣列
        return xx

    def getFitness(self, xx): #計算fitness，為流程時間最小
        meanFi = np.empty((xx.shape[0],), dtype=np.float32) #創造空矩陣來存放每列jobs排序的'mean總流程時間'
        for i, x_coord in enumerate(xx):
            Fi = np.cumsum(x_coord)
            meanFi[i] = Fi.sum()/jobs
        fitness = np.exp(self.DNAsize*2 / meanFi)
        return fitness, meanFi

    def select(self, fitness):
        idx = np.random.choice(np.arange(self.pop_size), size=self.pop_size, replace=True, p=fitness / fitness.sum())
        return self.pop[idx]
        #從pop_size值取pop_size數量的一維陣列根據機率p值賦予機率值>>產生idx陣列(機率高的會重複出現)
        # return新的DNA (fitness高的自然就出現多次，俄羅斯輪盤法則)

    def crossOver(self, parent, pop):
        if np.random.rand() < self.cross_rate:
            number = np.random.randint(0, self.pop_size, size=1) #隨機從0到pop_size中產生一個整數值
            cross_points = np.random.randint(0, 2, self.DNAsize).astype(np.bool) #產生與DNA一樣大小的boolean陣列
            keep_job = parent[~cross_points] #(從parant)將T>>F, F>>T互換後把F>>T的元素(jobs編號)存到keep_job中。(所以keep_job存的是int)
            swap_job = pop[number, np.isin(pop[number].ravel(), keep_job, invert=True)]
            #(從pop)把np.isin裡面的pop[number]中有與keep_job元素(job編號)相同的數字為Ture，反之False，再將T>>F，F>>T
            #再把pop中第number染色體中將np.isin最後篩出True的元素從pop[number]對應挑出存入swap_job
            parent[:] = np.concatenate((keep_job, swap_job))
        return parent

    def mutate(self, child):
        for point in range(self.DNAsize):
            if np.random.rand() < self.mutate_rate:
                swap_point = np.random.randint(0, self.DNAsize) #隨機從0到DNAsize中產生一個整數值作為"突變點"
                swapA, swapB = child[point], child[swap_point] #swapA為第point個job編號，swapB為第swap_point個job編號
                child[point], child[swap_point] = swapB, swapA #上面swapA與B的job編號交換
        return child

    def evolve(self, fitness):
        pop = self.select(fitness) #從母體中選出Fitness最好的染色體排序
        pop_copy = pop.copy() #複製到pop_copy變數中
        for parent in pop:  #把該染色體做處理
            child = self.crossOver(parent, pop_copy) #交配
            child = self.mutate(child) #變異
            parent[:] = child #產生新小孩
        self.pop = pop

class MyJobSchedule(object):
    def __init__(self, n_jobs):
        self.jobs_position = np.random.rand(n_jobs, 1)*50

myGA = GA(DNAsize=jobs, cross_rate=crossRate, mutate_rate=mutateRate, pop_size=popSize)
mySeq = MyJobSchedule(jobs)
for generation in range(iterations):
    lx = myGA.translateDNA(myGA.pop, mySeq.jobs_position)
    fitness, meanFi = myGA.getFitness(lx)
    myGA.evolve(fitness)
    best_idx = np.argmax(fitness)
    print('Generation:', generation, '| Best-MeanFi: %.2f' % meanFi[best_idx],)