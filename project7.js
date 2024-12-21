// This function takes the translation and two rotation angles (in radians) as input arguments.
// The two rotations are applied around x and y axes.
// It returns the combined 4x4 transformation matrix as an array in column-major order.
// You can use the MatrixMult function defined in project5.html to multiply two 4x4 matrices in the same format.
function GetModelViewMatrix( translationX, translationY, translationZ, rotationX, rotationY )
{
	var xRotMat = [
		1, 0, 0, 0,
		0, Math.cos(rotationX), Math.sin(rotationX), 0,
		0, -1 * Math.sin(rotationX), Math.cos(rotationX), 0,
		0, 0, 0, 1
	];

	var yRotMat = [
		Math.cos(rotationY), 0, -1 * Math.sin(rotationY), 0,
		0, 1, 0, 0,
		Math.sin(rotationY), 0, Math.cos(rotationY), 0,
		0, 0, 0, 1
	];

	var translationMat = [
		1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 1, 0,
		translationX, translationY, translationZ, 1
	];

	// first get XY
	var XY = MatrixMult( xRotMat, yRotMat );

	// then do T * (XY)
	var mv = MatrixMult( translationMat, XY );
	
	return mv;
}


// [TO-DO] Complete the implementation of the following class.

class MeshDrawer
{
	// The constructor is a good place for taking care of the necessary initializations.
	constructor()
	{
		// Compile the shader program
		this.prog = InitShaderProgram( modelVS, modelFS );
		
		// Get the ids of the uniform variables in the shaders
		this.mvp = gl.getUniformLocation( this.prog, 'mvp' );
		this.mv = gl.getUniformLocation( this.prog, 'mv' );
		this.normMat = gl.getUniformLocation( this.prog, 'normMat' );
		this.lightDir = gl.getUniformLocation( this.prog, 'lightDir' );
		this.swapFlag = gl.getUniformLocation( this.prog, 'swapFlag' );
		this.sampler = gl.getUniformLocation( this.prog, 'tex' );
		this.texFlag = gl.getUniformLocation( this.prog, 'texFlag' );
		this.alpha = gl.getUniformLocation( this.prog, 'alpha' );
		
		// Get the ids of the vertex attributes in the shaders
		this.vertPos = gl.getAttribLocation( this.prog, 'pos' );
		this.vertNorms = gl.getAttribLocation( this.prog, 'norm' );
		
		// Create the buffer objects
		this.vertbuffer = gl.createBuffer();
		this.normbuffer = gl.createBuffer();

		// Initialize texture stuff
		this.vertTex = gl.getAttribLocation( this.prog, 'txc' );
		this.texbuffer = gl.createBuffer();
	}
	
	// This method is called every time the user opens an OBJ file.
	// The arguments of this function is an array of 3D vertex positions,
	// an array of 2D texture coordinates, and an array of vertex normals.
	// Every item in these arrays is a floating point value, representing one
	// coordinate of the vertex position or texture coordinate.
	// Every three consecutive elements in the vertPos array forms one vertex
	// position and every three consecutive vertex positions form a triangle.
	// Similarly, every two consecutive elements in the texCoords array
	// form the texture coordinate of a vertex and every three consecutive 
	// elements in the normals array form a vertex normal.
	// Note that this method can be called multiple times.
	setMesh( vertPos, texCoords, normals )
	{
		// Update the contents of the vertex buffer objects.
		
		this.numTriangles = vertPos.length / 3;

		gl.bindBuffer(gl.ARRAY_BUFFER, this.vertbuffer);
		gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(vertPos), gl.STATIC_DRAW);

		gl.bindBuffer(gl.ARRAY_BUFFER, this.normbuffer);
		gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(normals), gl.STATIC_DRAW);
		
		gl.bindBuffer(gl.ARRAY_BUFFER, this.texbuffer);
		gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(texCoords), gl.STATIC_DRAW);
		
		// set the texture attribute aka send the data to the vs
		gl.vertexAttribPointer( this.vertTex, 2, gl.FLOAT, false, 0, 0 );
		gl.enableVertexAttribArray( this.vertTex );
		// at this point the vertex shader has the texture attribute
	}
	
	// This method is called when the user changes the state of the
	// "Swap Y-Z Axes" checkbox. 
	// The argument is a boolean that indicates if the checkbox is checked.
	swapYZ( swap )
	{
		// update the swapFlag uniform variable
		gl.useProgram( this.prog );
		if (swap) {
			gl.uniform1i( this.swapFlag, 1);
		}
		else {
			gl.uniform1i( this.swapFlag, 0);
		}
	}
	
	// This method is called to draw the triangular mesh.
	// The arguments are the model-view-projection transformation matrixMVP,
	// the model-view transformation matrixMV, the same matrix returned
	// by the GetModelViewProjection function above, and the normal
	// transformation matrix, which is the inverse-transpose of matrixMV.
	draw( matrixMVP, matrixMV, matrixNormal )
	{
		gl.useProgram( this.prog );
		gl.enableVertexAttribArray( this.vertPos );
		gl.bindBuffer( gl.ARRAY_BUFFER, this.vertbuffer );
		gl.vertexAttribPointer( this.vertPos, 3, gl.FLOAT, false, 0, 0 );

		gl.enableVertexAttribArray( this.vertNorms );
		gl.bindBuffer( gl.ARRAY_BUFFER, this.normbuffer );
		// every three vals in the buffer defines one vertex
		gl.vertexAttribPointer( this.vertNorms, 3, gl.FLOAT, false, 0, 0 );

		
		// set the uniform variables
		gl.uniformMatrix4fv( this.mvp, false, matrixMVP );
		gl.uniformMatrix4fv( this.mv, false, matrixMV );
		gl.uniformMatrix3fv( this.normMat, false, matrixNormal );

		gl.drawArrays( gl.TRIANGLES, 0, this.numTriangles );
	}
	
	// This method is called to set the texture of the mesh.
	// The argument is an HTML IMG element containing the texture data.
	setTexture( img )
	{
		gl.useProgram( this.prog );
		// should this not start as one if the show texture box is unchecked??
		gl.uniform1i( this.texFlag, 1);
		
		// Bind the texture
		const mytex = gl.createTexture();
		gl.bindTexture(gl.TEXTURE_2D, mytex);
		// You can set the texture image data using the following command.
		gl.texImage2D( gl.TEXTURE_2D, 0, gl.RGB, gl.RGB, gl.UNSIGNED_BYTE, img );

		gl.generateMipmap(gl.TEXTURE_2D);

		gl.activeTexture(gl.TEXTURE0);

		// Now that we have a texture, set
		// some uniform parameter(s) of the fragment shader, so that it uses the texture. (aka uniform sampler2D tex)
		gl.uniform1i(this.sampler, 0);
	}
	
	// This method is called when the user changes the state of the
	// "Show Texture" checkbox. 
	// The argument is a boolean that indicates if the checkbox is checked.
	showTexture( show )
	{
		// update the texFlag uniform variable
		gl.useProgram( this.prog );
		if (show) {
			gl.uniform1i( this.texFlag, 1);
		}
		else {
			gl.uniform1i( this.texFlag, 0);
		}
	}
	
	// This method is called to set the incoming light direction
	// They are floats
	setLightDir( x, y, z )
	{
		// set the uniform parameter(s) of the fragment shader to specify the light direction.
		gl.useProgram( this.prog );
		gl.uniform3f( this.lightDir, x, y, z);
	}
	
	// This method is called to set the shininess of the material
	// It's an int
	setShininess( shininess )
	{
		// set the uniform parameter(s) of the fragment shader to specify the shininess (alpha).
		gl.useProgram( this.prog );
		gl.uniform1i( this.alpha, shininess);
	}
}

// Vertex shader source code
var modelVS = `
	attribute vec3 norm;
	varying vec3 vertnorm;
	varying vec4 viewPos;
	uniform mat4 mv;
	uniform mat3 normMat;

	attribute vec3 pos;
	attribute vec2 txc;
	uniform mat4 mvp;
	uniform int swapFlag;
	varying vec2 texCoord;
	void main()
	{
		if (swapFlag == 0) {
			gl_Position = mvp * vec4(pos,1);

			vertnorm = normMat * norm;
			viewPos = mv * vec4(pos,1);
		}
		else {
			gl_Position = mvp * vec4(pos.x, pos.z, pos.y, 1);

			vertnorm = normMat * vec3(norm.x, norm.z, norm.y);
			viewPos = mv * vec4(pos.x, pos.z, pos.y, 1);
		}
		
		texCoord = txc;
	}
`;
// Fragment shader source code
var modelFS = `
	precision mediump float;
	// for shading in here, we need the positions in the view space,
	// but the rasterizer requires positions in the CVV.
    // so since gl_position is the input to the rasterizer, those should be in CVV (use mvp),
    // and then have varying variable with them in view space (use mv), so the fs can use those positions

	// shading requires the norm vect, light dir vect, and viewing dir vect all in view space (we'll calculate half ang vect)

	varying vec3 vertnorm;
	// this is already in view space
	uniform vec3 lightDir;
	// just need the positions in view space to be able to caluclate the viewing direction
	varying vec4 viewPos;
	uniform int alpha;
 
	uniform sampler2D tex;
	uniform int texFlag;
	varying vec2 texCoord;
	void main()
	{
		// dont forget to normalize all vects
  
		vec3 normal = normalize(vertnorm);
		vec3 lightDirection = normalize(lightDir);
  
		// calculate the viewing direction from the position
		vec3 viewingDir = vec3(float(-1) * viewPos.x, float(-1) * viewPos.y, float(-1) * viewPos.z);
		vec3 viewingDirection = normalize(viewingDir);

		// calculate the half angle from v and w
		vec3 halfV = vec3(lightDirection.x + viewingDirection.x, lightDirection.y + viewingDirection.y, lightDirection.z + viewingDirection.z);
		vec3 halfVect = normalize(halfV);

		// calculate cosine of the angle between n and h
		float cosPhi = dot(normal, halfVect);
		cosPhi = max(float(0), cosPhi);
  
		// calculate geometry term  (n dot w)
		float geoTerm = dot(normal, lightDirection);
		geoTerm = max(float(0), geoTerm);

		float specularTerm = pow(cosPhi, float(alpha));
		
 
		if (texFlag == 0) {
			gl_FragColor.r = float(1);
			gl_FragColor.g = float(1);
			gl_FragColor.b = float(1);

			gl_FragColor.rgb *= geoTerm;

			// Ks should be (1, 1, 1) so can just ignore that term in the model
			// also since intensity is just 1, we can ignore it too and just add the specular term and be done
			gl_FragColor.r = min(float(1), gl_FragColor.r + specularTerm);
			gl_FragColor.g = min(float(1), gl_FragColor.g + specularTerm);
			gl_FragColor.b = min(float(1), gl_FragColor.b + specularTerm);
		}
		else {
			// intensity and Ks should still be (1, 1, 1)

			gl_FragColor = texture2D(tex, texCoord);
			gl_FragColor.rgb *= geoTerm;

			gl_FragColor.r = min(float(1), gl_FragColor.r + specularTerm);
			gl_FragColor.g = min(float(1), gl_FragColor.g + specularTerm);
			gl_FragColor.b = min(float(1), gl_FragColor.b + specularTerm);
		}
	}
`;






// This function is called for every step of the simulation.
// Its job is to advance the simulation for the given time step duration dt.
// It updates the given positions and velocities.
function SimTimeStep( dt, positions, velocities, springs, stiffness, damping, particleMass, gravity, restitution )
{
	var forces = Array( positions.length ); // The total for per particle

	// Compute the total force of each particle
	for (var i = 0; i < forces.length; i++) {
		// add the gravity force
		forces[i] = gravity.mul(particleMass);
	}

	for (var i = 0; i < springs.length; i++) {
		// compute spring force:       Fs = k(l - lrest)d
		var p0Index = springs[i].p0;
		var p1Index = springs[i].p1;

		var len = ( positions[p1Index].sub(positions[p0Index]) ).len();
		var springDir = ( positions[p1Index].sub(positions[p0Index]) ).div(len);

		var F0s = springDir.mul( stiffness * (len - springs[i].rest) );

		// compute damping force:       Fd = kld
		var lenDeriv = springDir.dot( velocities[p1Index].sub(velocities[p0Index]) );

		var F0d = springDir.mul( damping * lenDeriv );

		var totalSpringForce = F0s.add(F0d);

		forces[p0Index].set( forces[p0Index].add(totalSpringForce) );
		forces[p1Index].set( forces[p1Index].sub(totalSpringForce) );
	}

	// Update positions and velocities
	// loop through the forces (1 for each particle) and calculate a from there  (a = f / m)
	for (var i = 0; i < forces.length; i++) {
		var a = forces[i].div(particleMass);

		velocities[i].set( velocities[i].add(a.mul(dt)) );
		positions[i].set( positions[i].add(velocities[i].mul(dt)) );


		// Handle collisions
		// check that x, y, and z are bw -1 and 1
		if (positions[i].x < -1.0) {
			var error = -1.0 - positions[i].x;
			var correction = restitution * error;
	
			positions[i].x = positions[i].x + error + correction;
			velocities[i].x = velocities[i].x * restitution * -1.0;
		}
	
		else if (positions[i].x > 1.0) {
			var error = positions[i].x - 1.0;
			var correction = restitution * error;
	
			positions[i].x = positions[i].x - error - correction;
			velocities[i].x = velocities[i].x * restitution * -1.0;
		}
		
	
		if (positions[i].y < -1.0) {
			var error = -1.0 - positions[i].y;
			var correction = restitution * error;
	
			positions[i].y = positions[i].y + error + correction;
			velocities[i].y = velocities[i].y * restitution * -1.0;
		}
	
		else if (positions[i].y > 1.0) {
			var error = positions[i].y - 1.0;
			var correction = restitution * error;
	
			positions[i].y = positions[i].y - error - correction;
			velocities[i].y = velocities[i].y * restitution * -1.0;
		}
		
	
		if (positions[i].z < -1.0) {
			var error = -1.0 - positions[i].z;
			var correction = restitution * error;
	
			positions[i].z = positions[i].z + error + correction;
			velocities[i].z = velocities[i].z * restitution * -1.0;
		}
	
		else if (positions[i].z > 1.0) {
			var error = positions[i].z - 1.0;
			var correction = restitution * error;
	
			positions[i].z = positions[i].z - error - correction;
			velocities[i].z = velocities[i].z * restitution * -1.0;
		}
	}

}

