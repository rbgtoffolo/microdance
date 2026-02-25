(
~isFirst = true;
~arraySynth = Array.fill(17, { | i |

	var freq = Scale.chromatic24.degreeToFreq((0..23).choose, 440, [i].linlin(0, 16, 5, 1));
	Synth(\default, [\freq, freq, \gate, 0])});


~arraySynth[3] = nil;
~arraySynth[4] = nil;
~width = 640;
~height = 480;
~diagonal = hypot(~width, ~height);

OSCdef(\setup, {
	| msg |
	~width = msg[2];
	~height = msg[3];
	~diagonal = hypot(~width-1, ~height-1);


}, \start);

OSCdef(\poses, {

	| msg |

	var pairs = msg.drop(1).clump(2);
	if (~isFirst and: { pairs.size == 17 }, {
		~firstPosition = pairs;
		pairs.size.do({ | i |
			~arraySynth[i].set(\gate, 1);
		});
		~isFirst = false;
	},{});

	pairs.do({ | pbody , i |
		var dist;
		if ((pbody[0] > 0 && pbody[1]>0), {
		dist = hypot(pbody[0] - ~firstPosition[i][0],pbody[1] - ~firstPosition[i][1])/ ~diagonal;

			~arraySynth[i].set(\amp, dist, \detune, dist.linlin(0,1.0,-100,100));},{});
	});


}, '\poses');

)
