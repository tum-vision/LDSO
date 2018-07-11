#pragma once
#ifndef LDSO_INDEX_THREAD_REDUCE_H_
#define LDSO_INDEX_THREAD_REDUCE_H_

#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>

#include "Settings.h"

using namespace std;
using namespace std::placeholders;

namespace ldso {

    namespace internal {

        /**
         * Multi thread tasks
         * use reduce function to multi threads a given task
         * like removing outliers or activating points
         * @tparam Running
         */
        template<typename Running>
        class IndexThreadReduce {

        public:
            EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

            inline IndexThreadReduce() {
                callPerIndex = bind(&IndexThreadReduce::callPerIndexDefault, this, _1, _2, _3, _4);
                for (int i = 0; i < NUM_THREADS; i++) {
                    isDone[i] = false;
                    gotOne[i] = true;
                    workerThreads[i] = thread(&IndexThreadReduce::workerLoop, this, i);
                }

            }

            inline ~IndexThreadReduce() {
                running = false;

                exMutex.lock();
                todo_signal.notify_all();
                exMutex.unlock();

                for (int i = 0; i < NUM_THREADS; i++)
                    workerThreads[i].join();


                printf("destroyed ThreadReduce\n");

            }

            inline void
            reduce(function<void(int, int, Running *, int)> callPerIndex, int first, int end, int stepSize = 0) {

                memset(&stats, 0, sizeof(Running));

                if (stepSize == 0)
                    stepSize = ((end - first) + NUM_THREADS - 1) / NUM_THREADS;

                unique_lock<mutex> lock(exMutex);

                // save
                this->callPerIndex = callPerIndex;
                nextIndex = first;
                maxIndex = end;
                this->stepSize = stepSize;

                // go worker threads!
                for (int i = 0; i < NUM_THREADS; i++) {
                    isDone[i] = false;
                    gotOne[i] = false;
                }

                // let them start!
                todo_signal.notify_all();


                // wait for all worker threads to signal they are done.
                while (true) {
                    // wait for at least one to finish
                    done_signal.wait(lock);

                    // check if actually all are finished.
                    bool allDone = true;
                    for (int i = 0; i < NUM_THREADS; i++)
                        allDone = allDone && isDone[i];

                    // all are finished! exit.
                    if (allDone)
                        break;
                }

                nextIndex = 0;
                maxIndex = 0;
                this->callPerIndex = bind(&IndexThreadReduce::callPerIndexDefault, this, _1, _2, _3, _4);
            }

            Running stats;

        private:
            thread workerThreads[NUM_THREADS];
            bool isDone[NUM_THREADS];
            bool gotOne[NUM_THREADS];

            mutex exMutex;
            condition_variable todo_signal;
            condition_variable done_signal;

            int nextIndex =0;
            int maxIndex =0;
            int stepSize =1;

            bool running =true;

            function<void(int, int, Running *, int)> callPerIndex;

            void callPerIndexDefault(int i, int j, Running *k, int tid) {
                printf("ERROR: should never be called....\n");
                assert(false);
            }

            void workerLoop(int idx) {
                unique_lock<mutex> lock(exMutex);

                while (running) {
                    // try to get something to do.
                    int todo = 0;
                    bool gotSomething = false;
                    if (nextIndex < maxIndex) {
                        // got something!
                        todo = nextIndex;
                        nextIndex += stepSize;
                        gotSomething = true;
                    }

                    // if got something: do it (unlock in the meantime)
                    if (gotSomething) {
                        lock.unlock();

                        assert(callPerIndex != 0);

                        Running s;
                        memset(&s, 0, sizeof(Running));
                        callPerIndex(todo, std::min(todo + stepSize, maxIndex), &s, idx);
                        gotOne[idx] = true;
                        lock.lock();
                        stats += s;
                    } // otherwise wait on signal, releasing lock in the meantime.
                    else {
                        if (!gotOne[idx]) {
                            lock.unlock();
                            assert(callPerIndex != 0);
                            Running s;
                            memset(&s, 0, sizeof(Running));
                            callPerIndex(0, 0, &s, idx);
                            gotOne[idx] = true;
                            lock.lock();
                            stats += s;
                        }
                        isDone[idx] = true;
                        done_signal.notify_all();
                        todo_signal.wait(lock);
                    }
                }
            }
        };

    }
}

#endif // LDSO_INDEX_THREAD_REDUCE_H_
