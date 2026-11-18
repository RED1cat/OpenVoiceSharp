using System;
using System.Diagnostics;

namespace OpenVoiceSharp
{
    public enum RecommendedChunkAmount
    {
        Unity = 18,
    }

    /// <summary>
    /// A thread-safe circular buffer of audio chunks (frames).
    /// Data can be read either entirely by filling or partially (the remainder),
    /// and data that is "stuck" due to network latency can be reset by timeout (<see cref="TryFlushStale"/>).
    /// </summary>
    /// <typeparam name="T">Byte/short/float depending on the task.</typeparam>
    public sealed class CircularAudioBuffer<T> where T : struct
    {
        private readonly object Lock = new();
        private readonly T[] Buffer;

        private int Head;
        private int Count;
        private long LastPushTimestamp = Stopwatch.GetTimestamp();

        /// <summary>The length of the buffer in samples.</summary>
        public int BufferLength { get; }
        /// <summary>The length of one chunk in samples.</summary>
        public int ChunkSize { get; }
        /// <summary>The maximum number of chunks.</summary>
        public int ChunkCapacity { get; }
        public int ChunksAvailable { get { lock (Lock) return Count; } }
        public int BufferAvailable => ChunksAvailable * ChunkSize;
        public bool BufferFull { get { lock (Lock) return Count == ChunkCapacity; } }
        public bool CanReadChunk => ChunksAvailable > 0;
        public bool HasData => ChunksAvailable > 0;

        /// <summary>How many milliseconds have passed since the last successful PushChunk.</summary>
        public double MillisecondsSinceLastPush
        {
            get
            {
                lock (Lock)
                    return (Stopwatch.GetTimestamp() - LastPushTimestamp) * 1000.0 / Stopwatch.Frequency;
            }
        }

        private void CopyOut(int chunks, T[] dest, int destOffset)
        {
            for (int i = 0; i < chunks; i++)
            {
                int src = ((Head + i) % ChunkCapacity) * ChunkSize;
                Array.Copy(Buffer, src, dest, destOffset + i * ChunkSize, ChunkSize);
            }
            Head = (Head + chunks) % ChunkCapacity;
            Count -= chunks;
        }

        /// <summary>
        /// Reads the oldest chunk.
        /// </summary>
        public T[] ReadChunk()
        {
            lock (Lock)
            {
                if (Count == 0)
                    throw new InvalidOperationException("No chunks are available.");

                T[] chunk = new T[ChunkSize];
                CopyOut(1, chunk, 0);
                return chunk;
            }
        }

        /// <summary>
        /// Trying to read the oldest chunk.
        /// </summary>
        public bool TryReadChunk(out T[] chunk)
        {
            lock (Lock)
            {
                if (Count == 0)
                {
                    chunk = Array.Empty<T>();
                    return false;
                }

                chunk = new T[ChunkSize];
                CopyOut(1, chunk, 0);
                return true;
            }
        }

        public void ReadChunkTo(T[] target, int offset = 0) => ReadChunk().CopyTo(target, offset);

        /// <summary>
        /// Reads EVERYTHING that is currently in the buffer (even if it is not full) and clears it.
        /// </summary>
        public T[] ReadAllBuffer()
        {
            lock (Lock)
            {
                T[] result = new T[Count * ChunkSize];
                CopyOut(Count, result, 0);
                return result;
            }
        }

        /// <summary>
        /// Reads everything available in target and clears the buffer.
        /// </summary>
        /// <returns>The number of samples copied.</returns>
        public int ReadAllBufferTo(T[] target, int offset = 0)
        {
            lock (Lock)
            {
                int samples = Count * ChunkSize;
                if (target.Length - offset < samples)
                    throw new ArgumentException("Target buffer is too small.", nameof(target));

                CopyOut(Count, target, offset);
                return samples;
            }
        }

        /// <summary>
        /// Reads everything available and adds silence (default) to the full length of the buffer.
        /// Convenient for engines that need a fixed-size buffer.
        /// </summary>
        public T[] ReadAllBufferPadded()
        {
            lock (Lock)
            {
                T[] result = new T[BufferLength];
                CopyOut(Count, result, 0);
                return result;
            }
        }

        /// <summary>
        /// If there is still data in the buffer, but no new data has been received for longer <paramref name="timeoutMs"/>,
        /// takes this remainder and clears the buffer. Call it periodically from the playback loop.
        /// </summary>
        /// <param name="timeoutMs">How long to wait for new data (for example 3-5 * 20ms).</param>
        /// <param name="data">The rest of the data (an empty array if no reset is needed).</param>
        /// <param name="padToFullBuffer">Add silence to the full length of the buffer.</param>
        /// <returns>true if the remainder has been reset.</returns>
        public bool TryFlushStale(int timeoutMs, out T[] data, bool padToFullBuffer = false)
        {
            lock (Lock)
            {
                double elapsedMs = (Stopwatch.GetTimestamp() - LastPushTimestamp) * 1000.0 / Stopwatch.Frequency;

                if (Count == 0 || elapsedMs < timeoutMs)
                {
                    data = Array.Empty<T>();
                    return false;
                }

                data = new T[padToFullBuffer ? BufferLength : Count * ChunkSize];
                CopyOut(Count, data, 0);
                return true;
            }
        }

        /// <summary>
        /// Completely clears the buffer.
        /// </summary>
        public void Clear()
        {
            lock (Lock)
            {
                Head = 0;
                Count = 0;
            }
        }

        /// <summary>
        /// Adds a chunk to the buffer.
        /// </summary>
        /// <param name="chunk">Chunk/frame.</param>
        /// <param name="overwriteOldest">
        /// If the buffer is full: true - discard the oldest chunk and write a new one,
        /// false - ignore the new chunk.
        /// </param>
        /// <returns>true if the chunk is written.</returns>
        public bool PushChunk(T[] chunk, bool overwriteOldest = false)
        {
            if (chunk.Length != ChunkSize)
                throw new ArgumentException($"Invalid chunk size. Submitted {chunk.Length} - should be {ChunkSize}", nameof(chunk));

            lock (Lock)
            {
                if (Count == ChunkCapacity)
                {
                    if (!overwriteOldest) return false;

                    Head = (Head + 1) % ChunkCapacity;
                    Count--;
                }

                int tail = (Head + Count) % ChunkCapacity;
                Array.Copy(chunk, 0, Buffer, tail * ChunkSize, ChunkSize);
                Count++;

                LastPushTimestamp = Stopwatch.GetTimestamp();
                return true;
            }
        }

        /// <summary>
        /// Creates a circular audio buffer.
        /// </summary>
        /// <param name="chunkSize">The size of a SINGLE frame in samples (see VoiceUtilities).</param>
        /// <param name="amountOfChunks">The number of chunks. Bigger = more stable, but higher latency (20 * amountOfChunks ms).</param>
        public CircularAudioBuffer(int chunkSize, int amountOfChunks = 18)
        {
            if (chunkSize <= 0) throw new ArgumentOutOfRangeException(nameof(chunkSize));
            if (amountOfChunks <= 0) throw new ArgumentOutOfRangeException(nameof(amountOfChunks));

            ChunkSize = chunkSize;
            ChunkCapacity = amountOfChunks;
            BufferLength = chunkSize * amountOfChunks;
            Buffer = new T[BufferLength];
        }

        public CircularAudioBuffer(int chunkSize, RecommendedChunkAmount engine)
            : this(chunkSize, (int)engine) { }
    }
}