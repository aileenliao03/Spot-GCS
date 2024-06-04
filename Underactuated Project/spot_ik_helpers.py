import numpy as np
from math import cos, sin, pi, atan2, sqrt

############################################################
##################### Transformations ######################
############################################################

def rotx(ang):
    """Create a 3x3 numpy rotation matrix about the x axis

    The three columns represent the new basis vectors in the global coordinate
    system of a coordinate system rotated by this matrix.

    Args:
        ang: angle for rotation in radians
    
    Returns:
        The 3D rotation matrix about the x axis
    """
    rotxMatrix = np.array(
        [   [1,             0,              0],
            [0,      cos(ang),      -sin(ang)],
            [0,      sin(ang),       cos(ang)]  ])

    return rotxMatrix


def roty(ang):
    """Create a 3x3 numpy rotation matrix about the y axis

    The three columns represent the new basis vectors in the global coordinate
    system of a coordinate system rotated by this matrix.

    Args:
        ang: angle for rotation in radians
    
    Returns:
        The 3D rotation matrix about the y axis
    """
    rotyMatrix = np.array(
        [   [ cos(ang),      0,       sin(ang)],
            [       0,       1,              0],
            [-sin(ang),      0,       cos(ang)]  ])

    return rotyMatrix


def rotz(ang):
    """Create a 3x3 numpy rotation matrix about the z axis

    The three columns represent the new basis vectors in the global coordinate
    system of a coordinate system rotated by this matrix.

    Args:
        ang: angle for rotation in radians
    
    Returns:
        The 3D rotation matrix about the z axis
    """
    rotzMatrix = np.array(
        [   [cos(ang),   -sin(ang),             0],
            [sin(ang),    cos(ang),             0],
            [       0,           0,             1]  ])

    return rotzMatrix

def rotxyz(x_ang,y_ang,z_ang):
    """Creates a 3x3 numpy rotation matrix from three rotations done in the order
    of x, y, and z in the local coordinate frame as it rotates.

    The three columns represent the new basis vectors in the global coordinate
    system of a coordinate system rotated by this matrix.

    Args:
        x_ang: angle for rotation about the x axis in radians
        y_ang: angle for rotation about the y axis in radians
        z_ang: angle for rotation about the z axis in radians

    Returns:
        The 3D rotation matrix for a x, y, z rotation
    """
    # return rotx(x_ang) @ roty(y_ang) @ rotz(z_ang)
    return np.matmul(np.matmul(rotx(x_ang), roty(y_ang)), rotz(z_ang))


def homog_rotxyz(x_ang,y_ang,z_ang):
    """Creates a 4x4 numpy homogeneous rotation matrix from three rotations
    done in the order x, y, and z in the local coordinate frame as it rotates. This is
    the same as the output of homog_trans except with no translation

    The three columns and rows represent the new basis vectors in the global coordinate
    system of a coordinate system rotated by this matrix.

    Args:
        x_ang: angle for rotation about the x axis in radians
        y_ang: angle for rotation about the y axis in radians
        z_ang: angle for rotation about the z axis in radians

    Returns:
        The homogenous transformation matrix for a x, y, z rotation and translation
    """
    return np.block( [  [rotxyz(x_ang,y_ang,z_ang), np.array([[0],[0],[0]])], [np.array([0,0,0,1])]  ]  )

def homog_transxyz(x,y,z):
    """Creates a 4x4 numpy linear transformation matrix

    Args:
        x: translation in x
        y: translation in y
        z: translation in z

    Returns:
        4x4 numpy array for a linear translation on a 4x1 vector
    """

    return np.block([ [np.eye(3,3), np.array([[x],[y],[z]]) ],  [np.array([0,0,0,1])]  ]   )

def homog_transform(x_ang,y_ang,z_ang,x_t,y_t,z_t):
    """Creates a 4x4 numpy rotation and transformation matrix from three rotations
    done in the order x, y, and z in the local coordinate frame as it rotates, then 
    a transformation in x, y, and z in that rotate coordinate frame.

    The three columns and rows represent the new basis vectors in the global coordinate
    system of a coordinate system rotated by this matrix. The last column and three rows
    represents the translation in the rotated coordinate frame

    Args:
        x_ang: angle for rotation about the x axis in radians
        y_ang: angle for rotation about the y axis in radians
        z_ang: angle for rotation about the z axis in radians
        x_t: linear translation in x
        y_t: linear translation in y
        z_t: linear translation in z

    Returns:
        The homogenous transformation matrix for a x, y, z rotation and translation
    """
    # return homog_rotxyz(x_ang,y_ang,z_ang) @ homog_transxyz(x_t,y_t,z_t)
    return np.matmul(homog_rotxyz(x_ang,y_ang,z_ang), homog_transxyz(x_t,y_t,z_t))

def ht_inverse(ht):
    '''Calculate the inverse of a homogeneous transformation matrix

    The inverse of a homogeneous transformation matrix can be represented as a
    a matrix product of the following:

                -------------------   ------------------- 
                |           |  0  |   | 1   0   0  -x_t |
    ht_inv   =  |   R^-1    |  0  | * | 0   1   0  -y_t |
                |___________|  0  |   | 0   0   1  -z_t |
                | 0   0   0 |  1  |   | 0   0   0   1   |
                -------------------   -------------------

    Where R^-1 is the ivnerse of the rotation matrix portion of the homogeneous
    transform (the first three rows and columns). Note that the inverse
    of a rotation matrix is equal to its transpose. And x_t, y_t, z_t are the
    linear trasnformation portions of the original transform.    
    
    Args
        ht: Input 4x4 nump matrix homogeneous transformation

    Returns:
        A 4x4 numpy matrix that is the inverse of the inputted transformation
    '''
    # Get the rotation matrix part of the homogeneous transform and take the transpose to get the inverse
    temp_rot = ht[0:3,0:3].transpose()

    # Get the linear transformation portion of the transform, and multiply elements by -1
    temp_vec = -1*ht[0:3,3]

    # Block the inverted rotation matrix back to a 4x4 homogeneous transform matrix
    temp_rot_ht = np.block([ [temp_rot            ,   np.zeros((3,1))],
                             [np.zeros((1,3))     ,         np.eye(1)] ])

    # Create a linear translation homogeneous transformation matrix 
    temp_vec_ht = np.eye(4)
    temp_vec_ht[0:3,3] = temp_vec

    # Return the matrix product
    # return temp_rot_ht @ temp_vec_ht
    return np.matmul(temp_rot_ht, temp_vec_ht)


#######################################################################
################## Forward and Inverse Kinematics #####################
#######################################################################

def t_rightback(t_m,l,w):
    '''Creates a 4x4 numpy homogeneous transformation matrix representing coordinate system and 
    position of the rightback leg of a quadriped. Assumes legs postioned in corners of a rectangular
    plane defined by a width and length 

    Args:
        t_m: 4x4 numpy matrix. Homogeneous transform representing the coordinate system of the center
        of the robot body
        l: length of the robot body
        w: width of the robot body

    Returns: 
        4x4 numpy matrix. A homogeneous transformation representing the position of the right back leg
    '''
    temp_homog_transf = np.block( [ [ roty(pi/2), np.array([[-l/2],[0],[w/2]])  ],
                                    [np.array([0,0,0,1])] ]    )
    # return t_m @ temp_homog_transf
    return np.matmul(t_m,temp_homog_transf)

def t_rightfront(t_m,l,w):
    '''Creates a 4x4 numpy homogeneous transformation matrix representing coordinate system and 
    position of the rightfront leg of a quadriped. Assumes legs postioned in corners of a rectangular
    plane defined by a width and length 

    Args:
        t_m: 4x4 numpy matrix. Homogeneous transform representing the coordinate system of the center
        of the robot body
        l: length of the robot body
        w: width of the robot body

    Returns: 
        4x4 numpy matrix. A homogeneous transformation representing the position of the right front leg
    '''
    temp_homog_transf = np.block( [ [ roty(pi/2), np.array([[l/2],[0],[w/2]])  ],
                                    [np.array([0,0,0,1])] ]    )
    # return t_m @ temp_homog_transf
    return np.matmul(t_m,temp_homog_transf)

def t_leftfront(t_m,l,w):
    '''Creates a 4x4 numpy homogeneous transformation matrix representing coordinate system and 
    position of the left front leg of a quadriped. Assumes legs postioned in corners of a rectangular
    plane defined by a width and length 

    Args:
        t_m: 4x4 numpy matrix. Homogeneous transform representing the coordinate system of the center
        of the robot body
        l: length of the robot body
        w: width of the robot body

    Returns: 
        4x4 numpy matrix. A homogeneous transformation representing the position of the left front leg
    '''
    temp_homog_transf = np.block( [ [ roty(-pi/2), np.array([[l/2],[0],[-w/2]])  ],
                                    [np.array([0,0,0,1])] ]    )
    # return t_m @ temp_homog_transf
    return np.matmul(t_m,temp_homog_transf)

def t_leftback(t_m,l,w):
    '''Creates a 4x4 numpy homogeneous transformation matrix representing coordinate system and 
    position of the left back leg of a quadriped. Assumes legs postioned in corners of a rectangular
    plane defined by a width and length 

    Args:
        t_m: 4x4 numpy matrix. Homogeneous transform representing the coordinate system of the center
        of the robot body
        l: length of the robot body
        w: width of the robot body

    Returns: 
        4x4 numpy matrix. A homogeneous transformation representing the position of the left back leg
    '''
    temp_homog_transf = np.block( [ [ roty(-pi/2), np.array([[-l/2],[0],[-w/2]])  ],
                                    [np.array([0,0,0,1])] ]    )
    # return t_m @ temp_homog_transf
    return np.matmul(t_m,temp_homog_transf)


def t_0_to_1(theta1,l1):
    '''Create the homogeneous transformation matrix for joint 0 to 1 for a quadriped leg.

    Args:
        theta1: Rotation angle in radians of the hip joint
        l1: Length of the hip joint link

    Returns:
        A 4x4 numpy matrix. Homogeneous transform from joint 0 to 1
    '''
    # I believe there is a typo in the paper. The paper lists this matrix as:
    # 
    # t =    [ cos(theta1)     -sin(theta1)    0        -l1*cos(theta1);
    #          sin(theta1)     -cos(theta1)    0        -l1*sin(theta1);
    #                    1                0    0                      0;
    #                    0                0    0                      1;]
    # 
    # However I believe index [2],[0] should be zero, and index [2],[2] should be 1 instead.
    # If not, then the rotated z axis disapears?? And from the diagram, it appears the transformed
    # axis's z axis is the same as the original. So I think the matrix should be:
    # 
    # t =    [ cos(theta1)     -sin(theta1)    0        -l1*cos(theta1);
    #          sin(theta1)     -cos(theta1)    0        -l1*sin(theta1);
    #                    0                0    1                      0;d
    #                    0                0    0                      1;]
    
    t_01 = np.block( [ [ rotz(theta1), np.array([[-l1*cos(theta1)],[-l1*sin(theta1)],[0]])  ],
                                    [np.array([0,0,0,1])] ]    )
    return t_01


def t_1_to_2():
    '''Create the homogeneous transformation matrix for joint 1 to 2 for a quadriped leg.

    Args:
        None

    Returns:
        A 4x4 numpy matrix. Homogeneous transform from joint 1 to 2
    '''
    # I believe there is a typo in the paper. The paper lists this matrix as:
    # 
    # t =    [           0                0        -1                      0;
    #                   -1                0         0                      0;
    #                    0                0         1                      0;
    #                    0                0         0                      1;]
    # 
    # However I believe index [1],[2] should be 1, and index [2],[2] should be 0.
    # If not, then the rotated y axis disapears?? So I think the matrix should be:
    # 
    # t =    [           0                0        -1                      0;
    #                   -1                0         0                      0;
    #                    0                1         0                      0;
    #                    0                0         0                      1;]
    # 
    t_12 = np.array([[ 0,  0, -1,  0],
                     [-1,  0,  0,  0],
                     [ 0,  1,  0,  0],
                     [ 0,  0,  0,  1]])
    return t_12

def t_2_to_3(theta2,l2):
    '''Create the homogeneous transformation matrix for joint 1 to 2 for a quadriped leg.

    Args:
        theta2: Rotation angle in radians of the leg joint
        l2: Length of the upper leg link

    Returns:
        A 4x4 numpy matrix. Homogeneous transform from joint 2 to 3
    '''

    t_23 = np.block( [ [ rotz(theta2), np.array([[l2*cos(theta2)],[l2*sin(theta2)],[0]])  ],
                                    [np.array([0,0,0,1])] ]    )
    return t_23

def t_3_to_4(theta3,l3):
    '''Create the homogeneous transformation matrix for joint 3 to 4 for a quadriped leg.

    Args:
        theta3: Rotation angle in radians of the knee joint
        l3: Length of the lower leg link

    Returns:
        A 4x4 numpy matrix. Homogeneous transform from joint 3 to 4
    '''

    t_34 = np.block( [ [ rotz(theta3), np.array([[l3*cos(theta3)],[l3*sin(theta3)],[0]])  ],
                                    [np.array([0,0,0,1])] ]    )
    return t_34

def ikine(x4,y4,z4,l1,l2,l3,legs12=True):
    '''Use inverse kinematics fo calculate the leg angles for a leg to achieve a desired
    leg end point position (x4,y4,z4)

    Args:
        x4: x position of leg end point relative to leg start point coordinate system.
        y4: y position of leg end point relative to leg start point coordinate system.
        z4: z position of leg end point relative to leg start point coordinate system.
        l1: leg link 1 length
        l2: leg link 2 length
        l3: leg link 3 length
        legs12: Optional input, boolean indicating whether equations are for legs 1 or 2. 
                If false, then equation for legs 3 and 4 is used

    Returns:
        A length 3 tuple of leg angles in the order (q1,q2,q3)
    '''

    # Supporting variable D
    D = (x4**2 + y4**2 + z4**2 - l1**2 - l2**2 - l3**2)/(2*l2*l3)

    if legs12 == True:
        q3 = atan2(sqrt(1-D**2),D)
    else:
        q3 = atan2(-sqrt(1-D**2),D)
    
    q2 = atan2(z4, sqrt(x4**2 + y4**2 - l1**2)) - atan2(l3*sin(q3), l2 + l3*cos(q3) )  

    # Following the equations from the paper there seems to be an error:
    # The first y4 in the equation below should not contain a minus sign (it does in the paper)
    q1 = - atan2(y4, x4) - atan2(sqrt(x4**2 + y4**2 - l1**2), -l1)

    return (q1,q2,q3)


#########################################################################
################## Spot Leg and Stick Figure Models #####################
#########################################################################

d2r = pi/180
r2d = 180/pi

class SpotLeg(object):
    '''Encapsulates a spot micro leg that consists of 3 links and 3 joint angles
    
    Attributes:
        _q1: Rotation angle in radians of hip joint
        _q2: Rotation angle in radians of upper leg joint
        _q3: Rotation angle in radians of lower leg joint
        _l1: Length of leg link 1 (i.e.: hip joint)
        _l2: Length of leg link 2 (i.e.: upper leg)
        _l3: Length of leg link 3 (i.e.: lower leg)
        _ht_leg: Homogeneous transformation matrix of leg starting 
                 position and coordinate system relative to robot body.
                 4x4 np matrix  
        _leg12: Boolean specifying whether leg is 1 or 2 (rightback or rightfront)
                or 3 or 4 (leftfront or leftback)  
    '''

    def __init__(self,q1,q2,q3,l1,l2,l3,ht_leg_start,leg12):
        '''Constructor'''
        self._q1 = q1
        self._q2 = q2
        self._q3 = q3
        self._l1 = l1
        self._l2 = l2
        self._l3 = l3
        self._ht_leg_start = ht_leg_start
        self._leg12 = leg12

        # Create homogeneous transformation matrices for each joint
        self._t01 = t_0_to_1(self._q1,self._l1)
        self._t12 = t_1_to_2()
        self._t23 = t_2_to_3(self._q2,self._l2)
        self._t34 = t_3_to_4(self._q3,self._l3)


    def set_angles(self,q1,q2,q3):
        '''Set the three leg angles and update transformation matrices as needed'''
        self._q1 = q1
        self._q2 = q2
        self._q3 = q3
        self._t01 = t_0_to_1(self._q1,self._l1)
        self._t23 = t_2_to_3(self._q2,self._l2)
        self._t34 = t_3_to_4(self._q3,self._l3)
    
    def set_homog_transf(self,ht_leg_start):
        '''Set the homogeneous transformation of the leg start position'''
        self._ht_leg_start = ht_leg_start

    def get_homog_transf(self):
        '''Return this leg's homogeneous transformation of the leg start position'''
        return self._ht_leg_start

    def set_foot_position_in_local_coords(self,x4,y4,z4):
        '''Set the position of the foot by computing joint angles via inverse kinematics from inputted coordinates.
        Leg's coordinate frame is the frame defined by self._ht_leg_start

        Args:
            x4: Desired foot x position in leg's coordinate frame
            y4: Desired foot y position in leg's coordinate frame
            z4: Desired foot z position in leg's coordinate frame
        Returns:
            Nothing
        '''
        # Run inverse kinematics and get joint angles
        leg_angs = ikine(x4,y4,z4,self._l1,self._l2,self._l3,self._leg12)

        # Call method to set joint angles for leg
        self.set_angles(leg_angs[0],leg_angs[1],leg_angs[2])

    def set_foot_position_in_global_coords(self,x4,y4,z4):
        ''' Set the position of the foot by computing joint angles via inverse kinematics from inputted coordinates.
        Inputted coordinates in the global coordinate frame

        Args:
            x4: Desired foot x position in global coordinate frame
            y4: Desired foot y position in global coordinate frame
            z4: Desired foot z position in global coordinate frame
        Returns:
            Nothing
        '''
        # Get inverse of leg's homogeneous transform
        ht_leg_inv = ht_inverse(self._ht_leg_start)

        # Convert the foot coordinates for use with homogeneous transforms, e.g.:
        # p4 = [x4, y4, z4, 1]
        p4_global_coord = np.block( [np.array([x4, y4, z4]), np.array([1])])

        # Calculate foot coordinates in each leg's coordinate system
        p4_in_leg_coords = ht_leg_inv.dot(p4_global_coord) 

        # Call this leg's position set function for coordinates in local frame
        self.set_foot_position_in_local_coords(p4_in_leg_coords[0],p4_in_leg_coords[1],p4_in_leg_coords[2])

    def get_leg_angles(self):
        '''Return leg angles as a tuple of 3 angles, (q1, q2, q3)'''
        return (self._q1,self._q2,self._q3)


class SpotStickFigure(object):
    """
    Encapuslates a 12 DOF spot stick figure. The 12 degrees of freedom represent the 
    twelve joint angles. Contains inverse kinematic capabilities
    
    Attributes:
        hip_length: Length of the hip joint
        upper_leg_length: length of the upper leg link
        lower_leg_length: length of the lower leg length
        body_width: width of the robot body
        body_height: length of the robot body

        x: x position of body center
        y: y position of body center
        z: z position of body center

        phi: roll angle in radians of body
        theta: pitch angle in radians of body
        psi: yaw angle in radians of body

        ht_body: homogeneous transformation matrix of the body

        rightback_leg_angles: length 3 list of joint angles. Order: hip, leg, knee
        rightfront_leg_angles: length 3 list of joint angles. Order: hip, leg, knee
        leftfront_leg_angles: length 3 list of joint angles. Order: hip, leg, knee
        leftback_leg_angles: length 3 list of joint angles. Order: hip, leg, knee

        leg_rightback
        leg_rightfront
        leg_leftfront
        leg_leftback
    """

    def __init__(self, x=0, z=0, psi=0):
        '''Constructor
        Allows specification of x,z position of body center (y: body height is fixed),
        as well as psi (yaw) orientation of body (roll, pitch fixed to 0).
        '''
        self.x = x
        self.y = 0.5 # fixed body height
        self.z = z
        
        self.phi = 0 # fixed roll
        self.theta = 0 # fixed pitch 
        self.psi = psi

        # Fixed body parameters for Spot
        self.hip_length = 0.11
        self.upper_leg_length = 0.321
        self.lower_leg_length = 0.3365
        self.body_width = 2 * 0.055
        self.body_length = 2 * 0.29785

        # Initialize Body Pose
        # Convention for this class is to initialize the body pose at a x,y,z position, with a phi,theta,psi orientation
        # To achieve this pose, need to apply a homogeneous translation first, then a homgeneous rotation
        # If done the other way around, a coordinate system will be rotate first, then translated along the rotated coordinate system
        # self.ht_body = homog_transxyz(self.x,self.y,self.z) @ homog_rotxyz(self.phi,self.psi,self.theta)
        self.ht_body = np.matmul(homog_transxyz(self.x,self.y,self.z), homog_rotxyz(self.phi,self.psi,self.theta))
        
        # Intialize all leg angles to 0, 30, 30 degrees
        self.rb_leg_angles   = [0,-30*d2r,60*d2r]
        self.rf_leg_angles   = [0,-30*d2r,60*d2r]
        self.lf_leg_angles   = [0,30*d2r,-60*d2r]
        self.lb_leg_angles   = [0,30*d2r,-60*d2r]

        # Create a dictionary to hold the legs of this spot micro object.
        # First initialize to empty dict
        self.legs = {}

        self.legs['leg_rightback'] = SpotLeg(self.rb_leg_angles[0],self.rb_leg_angles[1],self.rb_leg_angles[2],
                                             self.hip_length,self.upper_leg_length,self.lower_leg_length,
                                             t_rightback(self.ht_body,self.body_length,self.body_width),leg12=True) 
        
        self.legs['leg_rightfront'] = SpotLeg(self.rf_leg_angles[0],self.rf_leg_angles[1],self.rf_leg_angles[2],
                                              self.hip_length,self.upper_leg_length,self.lower_leg_length,
                                              t_rightfront(self.ht_body,self.body_length,self.body_width),leg12=True)
                                                  
        self.legs['leg_leftfront'] = SpotLeg(self.lf_leg_angles[0],self.lf_leg_angles[1],self.lf_leg_angles[2],
                                             self.hip_length,self.upper_leg_length,self.lower_leg_length,
                                             t_leftfront(self.ht_body,self.body_length,self.body_width),leg12=False)

        self.legs['leg_leftback'] = SpotLeg(self.lb_leg_angles[0],self.lb_leg_angles[1],self.lb_leg_angles[2],
                                            self.hip_length,self.upper_leg_length,self.lower_leg_length,
                                            t_leftback(self.ht_body,self.body_length,self.body_width),leg12=False) 

    def set_absolute_foot_coordinates(self,foot_coords):
        '''Set foot coordinates to a set inputted in the global coordinate frame and compute 
        and set the joint angles to achieve them using inverse kinematics
        
        Args:
            foot_coords: A 4x3 numpy matrix of desired (x4,y4,z4) positions for the end point (point 4) of each of
                    the four legs. I.e., the foot.
                    Leg order: rigthback, rightfront, leftfront, leftback. Example input:
                        np.array( [ [x4_rb,y4_rb,z4_rb],
                                    [x4_rf,y4_rf,z4_rf],
                                    [x4_lf,y4_lf,z4_lf],
                                    [x4_lb,y4_lb,z4_lb] ])
        Returns:
            Nothing
        '''

        # For each leg, call its method to set foot position in global coordinate frame
        
        foot_coords_dict = {'leg_rightback':foot_coords[0],
                            'leg_rightfront':foot_coords[1],
                            'leg_leftfront':foot_coords[2],
                            'leg_leftback':foot_coords[3]}
        
        for leg_name in self.legs:
            x4 = foot_coords_dict[leg_name][0]
            y4 = foot_coords_dict[leg_name][1]
            z4 = foot_coords_dict[leg_name][2]
            self.legs[leg_name].set_foot_position_in_global_coords(x4,y4,z4)

    def get_leg_angles(self):
        ''' Get the leg angles for all four legs
        Args:
            None
        Returns:
            leg_angs: Tuple of 4 of the leg angles. Legs in the order rightback
                      rightfront, leftfront, leftback. Angles in the order q1,q2,q3.
                      An example output:
                        ((rb_q1,rb_q2,rb_q3),
                         (rf_q1,rf_q2,rf_q3),
                         (lf_q1,lf_q2,lf_q3),
                         (lb_q1,lb_q2,lb_q3))
        '''
        return (    self.legs['leg_rightback'].get_leg_angles(),
                    self.legs['leg_rightfront'].get_leg_angles(),
                    self.legs['leg_leftfront'].get_leg_angles(),
                    self.legs['leg_leftback'].get_leg_angles()     )
